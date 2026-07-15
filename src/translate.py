"""
Smart routing...
Translation module — v3 (IndicTrans2 for Indic langs + restored pivot)

Routing (in priority order):
  1. same-language / empty text -> short-circuit, no model call.
  2. Both languages Indic, OR one side is English and the other Indic
     -> AI4Bharat IndicTrans2 directly (en-indic / indic-en / indic-indic).
  3. Neither language Indic, and a direct MarianMT model exists for the
     pair -> use it directly.
  4. English is one side and no direct MarianMT model exists -> try the
     Helsinki-NLP `opus-mt-{en}-{x}` / `opus-mt-{x}-{en}` naming
     convention as a fallback.
  5. EVERYTHING ELSE (no direct model on either side, e.g. es->fr,
     fr->de, and critically fr->hi / es->ta / de->bn etc.) -> pivot via
     English: source -> en -> target, two legs, each leg routed through
     whichever engine actually supports it (Marian for non-Indic legs,
     IndicTrans2 for Indic legs).

FIX (v3, from v2):
  v2 collapsed case 5 into a silent passthrough — `translated = text`,
  method "no_model" — for ANY non-Indic pair without a direct Marian
  model (e.g. es<->fr, which have no Helsinki-NLP direct checkpoint).
  This is why "Spanish -> French" was returning the original Spanish
  text untouched. The old pivot-via-English logic from translate_prev.py
  was dropped in the v2 rewrite and never re-added.

  v2 also had a second, related bug that hadn't been hit yet: any pair
  mixing an Indic language with a non-Indic, non-English language (e.g.
  fr->hi) was being routed straight into _run_indic(), which does
  INDIC_CODE_MAP[source_lang] — a KeyError for "fr", silently caught by
  the outer try/except and again returned as untranslated passthrough
  with method "indictrans2_failed".

  v3 restores pivoting for BOTH cases, but (unlike translate_prev.py,
  which assumed both pivot legs were always Marian) routes each leg
  through the correct engine: Marian for non-Indic<->en legs, IndicTrans2
  for Indic<->en legs. This is what makes fr->hi, es->ta, de->bn etc.
  work correctly instead of erroring out.

  Everything about the original IndicTrans2 integration (why it was
  chosen over Helsinki opus-mt-mul / opus-mt-dra for Indic languages,
  the quantization caveat, model/requirements notes) is unchanged from
  v2 — see below.

  Indic-language rationale (unchanged from v2):
      1. mul-en specifically has documented quality problems in the
         Indic->English direction (confirmed via community reports).
      2. Several of these language pairs' only real OPUS training data
         is drawn from religious text corpora (JW300), producing
         confidently wrong, unrelated translations for ordinary speech.
      3. Published research confirms INT8 quantization (which this
         module applies for speed) disproportionately degrades
         low-resource/Indic language quality specifically, sometimes to
         the point of losing translation ability entirely.
    IndicTrans2 exists specifically to solve this gap with curated,
    Indic-focused training data instead of generic multilingual corpora.

  - English<->French/German/Spanish/Chinese/Arabic pairs stay exactly as
    before on Helsinki-NLP MarianMT, since these were already verified
    working well and have no equivalent low-resource issue.

REQUIREMENTS (add to requirements.txt if not already present):
  IndicTransToolkit
  sentencepiece
  transformers==4.44.2       (IndicTransToolkit is incompatible with
                               newer transformers versions that removed
                               the transformers.onnx module)
  tokenizers==0.19.1         (compatible with transformers==4.44.2)
  huggingface_hub==0.26.5    (compatible with the above two; also used
                               directly now for the direct-model existence
                               check — already a transitive dependency of
                               transformers, so nothing new to install)

PREREQUISITE: IndicTransToolkit compiles a native extension on install.
On Windows this requires Microsoft Visual C++ Build Tools (the "Desktop
development with C++" workload) to be installed and visible on PATH
(verify with `where cl.exe` in the same terminal you run pip install
from — a fresh Developer PowerShell for VS is the most reliable way to
get this) BEFORE running `pip install -r requirements.txt`.

STILL WORTH RE-VERIFYING LOCALLY: the new pivot paths (fr->hi, es->ta,
de->bn, etc.) exercise a code path that has NOT been run end-to-end —
test each of these live before trusting them in production, same
caution as before with anything touching model downloads/native builds.
Note the pivot doubles latency (two model calls) and compounds error
for any pair that uses it, which is expected/inherent to pivoting, not
a bug — worth surfacing in the UI (e.g. via the `method` field) if
users care about translation quality on these specific pairs.
"""

from transformers import MarianMTModel, MarianTokenizer, AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import time
import traceback
from huggingface_hub import model_info as hf_model_info

# ─── MARIAN MT — European/CJK/Arabic pairs, unchanged from before ────
LANGUAGE_PAIRS = {
    ("en", "fr"): "Helsinki-NLP/opus-mt-en-fr",
    ("fr", "en"): "Helsinki-NLP/opus-mt-fr-en",
    ("en", "de"): "Helsinki-NLP/opus-mt-en-de",
    ("de", "en"): "Helsinki-NLP/opus-mt-de-en",
    ("en", "es"): "Helsinki-NLP/opus-mt-en-es",
    ("es", "en"): "Helsinki-NLP/opus-mt-es-en",
    ("en", "zh"): "Helsinki-NLP/opus-mt-en-zh",
    ("zh", "en"): "Helsinki-NLP/opus-mt-zh-en",
    ("en", "ar"): "Helsinki-NLP/opus-mt-en-ar",
    ("ar", "en"): "Helsinki-NLP/opus-mt-ar-en",

    # FIX: these non-English pairs were previously always pivoted through
    # English (2 model calls) even though Helsinki-NLP actually publishes
    # direct checkpoints for them — confirmed by checking the Hub
    # directly rather than assuming. One model call instead of two, and
    # no compounding pivot error for these specific pairs.
    ("fr", "de"): "Helsinki-NLP/opus-mt-fr-de",
    ("de", "fr"): "Helsinki-NLP/opus-mt-de-fr",
    ("fr", "es"): "Helsinki-NLP/opus-mt-fr-es",
    ("es", "fr"): "Helsinki-NLP/opus-mt-es-fr",
    ("de", "es"): "Helsinki-NLP/opus-mt-de-es",
    ("es", "de"): "Helsinki-NLP/opus-mt-es-de",
    ("ar", "fr"): "Helsinki-NLP/opus-mt-ar-fr",
    ("fr", "ar"): "Helsinki-NLP/opus-mt-fr-ar",
}

# ─── INDICTRANS2 — every Indic language, all pairs ─────────────
INDIC_LANGS = {"hi", "mr", "bn", "gu", "kn", "ml", "ur", "pa", "ta", "te"}

# App's 2-letter codes -> IndicTrans2's FLORES-style codes.
# Verify each against IndicTrans2's actual supported list before relying
# on it in production — transcribed from AI4Bharat's documentation.
INDIC_CODE_MAP = {
    "hi": "hin_Deva",
    "mr": "mar_Deva",
    "bn": "ben_Beng",
    "gu": "guj_Gujr",
    "kn": "kan_Knda",
    "ml": "mal_Mlym",
    "ur": "urd_Arab",
    "pa": "pan_Guru",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "en": "eng_Latn",
}

# Distilled checkpoints — smaller/faster than the 1B variants, chosen
# given this project's memory constraints (already at 16Gi on Cloud Run
# with OOM history — verify actual usage locally before deploying this).
INDICTRANS_MODELS = {
    "en-indic": "ai4bharat/indictrans2-en-indic-dist-200M",
    "indic-en": "ai4bharat/indictrans2-indic-en-dist-200M",
    # "indic-indic" checkpoint deliberately dropped — see translate()
    # below: Indic<->Indic now composes indic-en + en-indic instead of
    # calling the direct indic-indic checkpoint, which was the weaker of
    # the three and the suspected source of the X->hi asymmetry.
}


class Translator:
    def __init__(self):
        self._cache = {}          # MarianMT models
        self._indic_cache = {}    # IndicTrans2 models + tokenizers
        self._indic_processor = None
        self._marian_exists_cache = {}   # (src,tgt) -> bool, HF Hub lookups are cached, not repeated
        print("[Translator] Ready.")

    # ── MarianMT (non-Indic pairs) ─────────────────────────────
    def _load(self, model_name: str):
        if model_name in self._cache:
            return self._cache[model_name]
        print(f"[Translator] Loading {model_name}...")
        tok   = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        model.eval()
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        self._cache[model_name] = (tok, model)
        return tok, model

    def _run(self, text: str, model_name: str) -> str:
        tok, model = self._load(model_name)
        inputs = tok([text], return_tensors="pt",
                     padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            out = model.generate(**inputs, max_length=512,
                                  num_beams=1, do_sample=False)
        return tok.decode(out[0], skip_special_tokens=True)

    def _run_marian_pair(self, text: str, source_lang: str, target_lang: str) -> str:
        """Direct MarianMT call for a non-Indic pair, with the
        opus-mt-{src}-{tgt} naming fallback if not in LANGUAGE_PAIRS."""
        model_name = LANGUAGE_PAIRS.get((source_lang, target_lang))
        if not model_name:
            model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
        return self._run(text, model_name)

    def _direct_marian_model_exists(self, source_lang: str, target_lang: str) -> bool:
        """FIX: pivoting via English used to be assumed necessary for
        ANY non-Indic pair not hand-listed in LANGUAGE_PAIRS. But
        Helsinki-NLP publishes far more direct bilingual checkpoints than
        this file used to check for — hardcoding a full matrix by hand
        risks typos/wrong casing (their naming isn't perfectly
        consistent — e.g. some checkpoints use "opus-mt-de-ZH" not
        "opus-mt-de-zh") and will always lag behind what they actually
        publish. So: check the Hub directly (cheap metadata call, no
        model download) instead of guessing, and cache the result so
        it's only checked once per language pair for the life of this
        process, not once per translation call."""
        key = (source_lang, target_lang)
        if key in self._marian_exists_cache:
            return self._marian_exists_cache[key]

        model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
        try:
            hf_model_info(model_name)
            exists = True
        except Exception:
            exists = False

        self._marian_exists_cache[key] = exists
        print(f"[Translator] Direct model check {source_lang}->{target_lang}: "
              f"{'found' if exists else 'not found'} ({model_name})")
        return exists

    # ── IndicTrans2 (Indic pairs, and Indic<->English) ─────────
    def _get_indic_processor(self):
        if self._indic_processor is None:
            from IndicTransToolkit.processor import IndicProcessor
            self._indic_processor = IndicProcessor(inference=True)
        return self._indic_processor

    def _load_indic(self, direction: str):
        """direction: 'en-indic' | 'indic-en' | 'indic-indic'"""
        if direction in self._indic_cache:
            return self._indic_cache[direction]

        model_name = INDICTRANS_MODELS[direction]
        print(f"[Translator] Loading IndicTrans2 {direction} ({model_name})...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
        model.eval()
        # NOTE: quantization deliberately intact here for speed, BUT this
        # is worth re-testing against unquantized output if translation
        # quality still seems off — the earlier research on quantization
        # hurting Indic languages was about generic multilingual OPUS
        # models, not confirmed against IndicTrans2 specifically. Test
        # both if in doubt.
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        self._indic_cache[direction] = (tokenizer, model)
        return tokenizer, model

    def _run_indic(self, text: str, source_lang: str, target_lang: str) -> str:
        """source_lang/target_lang must each be 'en' or a key in
        INDIC_LANGS — do not call this with a third-party language on
        either side, it will KeyError on INDIC_CODE_MAP."""
        if source_lang == "en":
            direction = "en-indic"
        elif target_lang == "en":
            direction = "indic-en"
        else:
            # Every caller in this module pivots Indic<->Indic through
            # "en" (see translate()) rather than calling this directly
            # with two non-English Indic languages, so this should be
            # unreachable. Fail loudly rather than KeyError on the
            # removed "indic-indic" model entry if something changes
            # that assumption later.
            raise ValueError(
                f"_run_indic() called with two non-English languages "
                f"({source_lang}->{target_lang}) — pivot through 'en' instead, "
                f"the direct indic-indic checkpoint was intentionally dropped."
            )

        tokenizer, model = self._load_indic(direction)
        ip = self._get_indic_processor()

        src_code = INDIC_CODE_MAP[source_lang]
        tgt_code = INDIC_CODE_MAP[target_lang]

        batch = ip.preprocess_batch([text], src_lang=src_code, tgt_lang=tgt_code)
        inputs = tokenizer(
            batch, truncation=True, padding="longest",
            return_tensors="pt", max_length=256,
        )
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=256,
                # AI4Bharat's suggested default is num_beams=5. Lowered
                # to 1 here for latency parity with the rest of this
                # pipeline — raise it if quality matters more than speed
                # for your use case and you can afford slower turns.
                num_beams=1,
                num_return_sequences=1,
            )
        decoded = tokenizer.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        result = ip.postprocess_batch(decoded, lang=tgt_code)
        return result[0] if result else text

    # ── Pivot helpers — route each leg through whichever engine
    #    actually supports that language, not just Marian ──────
    def _to_english(self, text: str, source_lang: str) -> str:
        if source_lang == "en":
            return text
        if source_lang in INDIC_LANGS:
            return self._run_indic(text, source_lang, "en")
        return self._run_marian_pair(text, source_lang, "en")

    def _from_english(self, text: str, target_lang: str) -> str:
        if target_lang == "en":
            return text
        if target_lang in INDIC_LANGS:
            return self._run_indic(text, "en", target_lang)
        return self._run_marian_pair(text, "en", target_lang)

    # ── Public API (unchanged signature — no other files need edits) ──
    def translate(self, text: str, source_lang: str, target_lang: str) -> dict:
        if source_lang == target_lang:
            return {"translated_text": text, "source_lang": source_lang,
                     "target_lang": target_lang, "latency": 0.0, "method": "same"}
        if not text.strip():
            return {"translated_text": "", "source_lang": source_lang,
                     "target_lang": target_lang, "latency": 0.0, "method": "empty"}

        t0 = time.time()
        src_indic = source_lang in INDIC_LANGS
        tgt_indic = target_lang in INDIC_LANGS

        method = "direct"
        translated = text
        try:
            if source_lang == "en" and tgt_indic:
                translated = self._run_indic(text, source_lang, target_lang)
                method = "indictrans2"

            elif target_lang == "en" and src_indic:
                translated = self._run_indic(text, source_lang, target_lang)
                method = "indictrans2"

            elif src_indic and tgt_indic:
                # FIX: both Indic used to go straight through the direct
                # indic-indic checkpoint (ai4bharat/indictrans2-indic-indic-
                # dist-320M). That's the smallest/weakest of the three
                # IndicTrans2 checkpoints, and AI4Bharat's own benchmarks
                # show it noticeably behind composing the two directional
                # checkpoints — which matches the reported symptom exactly
                # (Hindi -> other Indic worked well since that's really
                # "en-indic" quality via a well-trained direction, while
                # other-Indic -> Hindi through the direct model was weak
                # or failing outright).
                #
                # Fix: never call the indic-indic checkpoint directly.
                # Always compose indic -> en -> indic using the two
                # directional checkpoints instead — both of which are
                # already loaded/cached for the en<->indic pairs elsewhere
                # in this app, so this doesn't even add a new model to
                # load. Costs a second forward pass (roughly 2x latency
                # for Indic<->Indic specifically), but that's the honest
                # trade for correctness here — test locally to confirm
                # this actually resolves it for you before assuming it's
                # fully fixed.
                print(f"[Translator] Indic<->Indic via en pivot: {source_lang} -> en -> {target_lang}")
                english_text = self._run_indic(text, source_lang, "en")
                translated   = self._run_indic(english_text, "en", target_lang)
                method = "indictrans2_pivot"

            elif not src_indic and not tgt_indic:
                # Neither side Indic -> MarianMT territory.
                if (source_lang, target_lang) in LANGUAGE_PAIRS:
                    translated = self._run_marian_pair(text, source_lang, target_lang)
                    method = "direct"
                elif source_lang == "en" or target_lang == "en":
                    # No exact entry but one side is English -> try the
                    # opus-mt-{en}-{x} / opus-mt-{x}-{en} naming fallback.
                    translated = self._run_marian_pair(text, source_lang, target_lang)
                    method = "fallback"
                elif self._direct_marian_model_exists(source_lang, target_lang):
                    # FIX: check before assuming pivot is needed — see
                    # _direct_marian_model_exists() for why this isn't
                    # just a hardcoded table.
                    translated = self._run_marian_pair(text, source_lang, target_lang)
                    method = "direct_discovered"
                else:
                    # Genuinely no direct model (confirmed via Hub check,
                    # not assumed) -> pivot.
                    print(f"[Translator] Pivoting (non-Indic): {source_lang} -> en -> {target_lang}")
                    english_text = self._to_english(text, source_lang)
                    translated = self._from_english(english_text, target_lang)
                    method = "pivot_via_english"

            else:
                # Exactly one side Indic, other side non-Indic and not
                # English (e.g. fr->hi, es->ta, de->bn) -> ALWAYS pivot
                # here, deliberately, even if a direct Helsinki-NLP
                # checkpoint happens to exist for this exact pair. Direct
                # non-English<->Indic checkpoints are generic multilingual
                # models trained on the same kind of sparse/low-quality
                # corpora (JW300 etc.) that caused this project to move
                # Indic languages to IndicTrans2 in the first place —
                # using one here would quietly reintroduce that problem
                # for the Indic leg. Pivoting through en (Marian for the
                # non-Indic leg, IndicTrans2 for the Indic leg) keeps
                # every Indic-involving translation on IndicTrans2.
                print(f"[Translator] Pivoting (mixed): {source_lang} -> en -> {target_lang}")
                english_text = self._to_english(text, source_lang)
                translated = self._from_english(english_text, target_lang)
                method = "pivot_via_english"

        except Exception as e:
            # FIX: str(e) alone was hiding the actual failure point (this
            # is exactly why the earlier X->hi bug report couldn't be
            # pinned down without a traceback) — print the full traceback
            # so a failure is diagnosable straight from the server log.
            print(f"[Translator] Translation failed ({source_lang}->{target_lang}): {e}")
            traceback.print_exc()
            translated = text
            method = "translation_failed"

        # FIX: flag suspicious output instead of returning it silently as
        # if it were a normal success — this is what let earlier passthrough
        # bugs (untranslated text returned as if translated) go unnoticed.
        if method not in ("translation_failed",):
            if not translated.strip():
                print(f"[Translator] WARNING: empty output for {source_lang}->{target_lang} (method={method})")
                method = f"{method}_empty_output"
            elif translated.strip() == text.strip():
                print(f"[Translator] WARNING: output identical to input for {source_lang}->{target_lang} "
                      f"(method={method}) — likely untranslated passthrough")
                method = f"{method}_unchanged"

        return {
            "translated_text": translated,
            "source_lang":     source_lang,
            "target_lang":     target_lang,
            "latency":         round(time.time() - t0, 2),
            "method":          method,
        }

    def preload(self, pairs: list):
        """
        Preload safely — skip any that fail without crashing server.
        Indic-involving pairs preload the relevant IndicTrans2 direction
        (only 3 possible directions total, so this stays fast even with
        many pairs listed — each direction's model loads once and is
        cached, subsequent pairs needing the same direction are instant).
        Pivot pairs (mixed Indic/non-Indic, or non-Indic pairs with no
        direct Marian model) preload BOTH legs so the first real request
        for that pair doesn't pay a cold-load penalty.
        """
        loaded  = 0
        skipped = 0
        for src, tgt in pairs:
            src_indic = src in INDIC_LANGS
            tgt_indic = tgt in INDIC_LANGS
            try:
                if src_indic and tgt_indic:
                    # FIX: no longer uses the direct indic-indic checkpoint
                    # (see translate() above) — preload both directional
                    # checkpoints it actually pivots through instead.
                    self._load_indic("indic-en")
                    self._load_indic("en-indic")
                    loaded += 1

                elif (src == "en" and tgt_indic) or (tgt == "en" and src_indic):
                    if src == "en":
                        self._load_indic("en-indic")
                    else:
                        self._load_indic("indic-en")
                    loaded += 1

                elif not src_indic and not tgt_indic:
                    if (src, tgt) in LANGUAGE_PAIRS:
                        self._load(LANGUAGE_PAIRS[(src, tgt)])
                        loaded += 1
                    elif src == "en" or tgt == "en":
                        self._load(f"Helsinki-NLP/opus-mt-{src}-{tgt}")
                        loaded += 1
                    elif self._direct_marian_model_exists(src, tgt):
                        # FIX: a direct model exists but wasn't in
                        # LANGUAGE_PAIRS — preload the one model it'll
                        # actually use, not two pivot legs it won't need.
                        self._load(f"Helsinki-NLP/opus-mt-{src}-{tgt}")
                        loaded += 1
                    else:
                        # confirmed no direct model -> preload both pivot legs
                        self._load(LANGUAGE_PAIRS.get((src, "en"), f"Helsinki-NLP/opus-mt-{src}-en"))
                        self._load(LANGUAGE_PAIRS.get(("en", tgt), f"Helsinki-NLP/opus-mt-en-{tgt}"))
                        loaded += 1

                else:
                    # mixed pivot: preload whichever legs are needed
                    if src_indic:
                        self._load_indic("indic-en")
                    else:
                        self._load(LANGUAGE_PAIRS.get((src, "en"), f"Helsinki-NLP/opus-mt-{src}-en"))
                    if tgt_indic:
                        self._load_indic("en-indic")
                    else:
                        self._load(LANGUAGE_PAIRS.get(("en", tgt), f"Helsinki-NLP/opus-mt-en-{tgt}"))
                    loaded += 1

            except Exception as e:
                print(f"[Translator] Skipping {src}→{tgt}: {e}")
                skipped += 1

        print(f"[Translator] Preload complete: {loaded} loaded, {skipped} skipped.")