"""
Translation module — v2 (IndicTrans2 for all Indic languages)

Routing:
  - ANY pair involving an Indic language (hi, mr, bn, gu, kn, ml, ur, pa,
    ta, te) -> AI4Bharat IndicTrans2. This includes direct Indic<->Indic
    translation with no English pivot, and replaces the earlier
    Helsinki-NLP opus-mt-en-dra / opus-mt-en-mul / opus-mt-mul-en
    approach entirely. Those models were dropped because:
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
  huggingface_hub==0.26.5    (compatible with the above two)

PREREQUISITE: IndicTransToolkit compiles a native extension on install.
On Windows this requires Microsoft Visual C++ Build Tools (the "Desktop
development with C++" workload) to be installed and visible on PATH
(verify with `where cl.exe` in the same terminal you run pip install
from — a fresh Developer PowerShell for VS is the most reliable way to
get this) BEFORE running `pip install -r requirements.txt`.

NOT YET RUN END-TO-END IN A LIVE ENVIRONMENT — test locally with real
text for each language pair before trusting this in production, per the
usual caution with anything touching model downloads/native builds.
"""

from transformers import MarianMTModel, MarianTokenizer, AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import time

# ─── MARIAN MT — European pairs only, unchanged from before ────
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
    "indic-indic": "ai4bharat/indictrans2-indic-indic-dist-320M",
}


class Translator:
    def __init__(self):
        self._cache = {}          # MarianMT models
        self._indic_cache = {}    # IndicTrans2 models + tokenizers
        self._indic_processor = None
        print("[Translator] Ready.")

    # ── MarianMT (European pairs) ─────────────────────────────
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

    # ── IndicTrans2 (all Indic pairs) ──────────────────────────
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
        # quality still seems off after switching to IndicTrans2 — the
        # earlier research on quantization hurting Indic languages was
        # about generic multilingual OPUS models, not confirmed against
        # IndicTrans2 specifically. Test both if in doubt.
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        self._indic_cache[direction] = (tokenizer, model)
        return tokenizer, model

    def _run_indic(self, text: str, source_lang: str, target_lang: str) -> str:
        if source_lang == "en":
            direction = "en-indic"
        elif target_lang == "en":
            direction = "indic-en"
        else:
            direction = "indic-indic"

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

    # ── Public API (unchanged signature — no other files need edits) ──
    def translate(self, text: str, source_lang: str, target_lang: str) -> dict:
        if source_lang == target_lang:
            return {"translated_text": text, "source_lang": source_lang,
                     "target_lang": target_lang, "latency": 0.0, "method": "same"}
        if not text.strip():
            return {"translated_text": "", "source_lang": source_lang,
                     "target_lang": target_lang, "latency": 0.0, "method": "empty"}

        t0 = time.time()
        involves_indic = source_lang in INDIC_LANGS or target_lang in INDIC_LANGS

        if involves_indic:
            try:
                translated = self._run_indic(text, source_lang, target_lang)
                method = "indictrans2"
            except Exception as e:
                print(f"[Translator] IndicTrans2 failed ({source_lang}->{target_lang}): {e}")
                translated = text
                method = "indictrans2_failed"

            return {
                "translated_text": translated,
                "source_lang":     source_lang,
                "target_lang":     target_lang,
                "latency":         round(time.time() - t0, 2),
                "method":          method,
            }

        # ── Non-Indic pairs: original MarianMT logic ──
        pair   = (source_lang, target_lang)
        model  = LANGUAGE_PAIRS.get(pair)
        method = "direct"

        if model:
            translated = self._run(text, model)
        elif source_lang == "en":
            method = "fallback"
            try:
                translated = self._run(text, f"Helsinki-NLP/opus-mt-en-{target_lang}")
            except Exception:
                translated = text
                method     = "no_model"
        elif target_lang == "en":
            method = "fallback"
            try:
                translated = self._run(text, f"Helsinki-NLP/opus-mt-{source_lang}-en")
            except Exception:
                translated = text
                method     = "no_model"
        else:
            method = "no_model"
            translated = text
            print(f"[Translator] No route for {source_lang}->{target_lang}")

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
        """
        loaded  = 0
        skipped = 0
        for src, tgt in pairs:
            if src in INDIC_LANGS or tgt in INDIC_LANGS:
                try:
                    if src == "en":
                        self._load_indic("en-indic")
                    elif tgt == "en":
                        self._load_indic("indic-en")
                    else:
                        self._load_indic("indic-indic")
                    loaded += 1
                except Exception as e:
                    print(f"[Translator] Skipping IndicTrans2 {src}→{tgt}: {e}")
                    skipped += 1
                continue

            m = LANGUAGE_PAIRS.get((src, tgt))
            if m:
                try:
                    self._load(m)
                    loaded += 1
                except Exception as e:
                    print(f"[Translator] Skipping {src}→{tgt} ({m}): {e}")
                    skipped += 1
        print(f"[Translator] Preload complete: {loaded} loaded, {skipped} skipped.")