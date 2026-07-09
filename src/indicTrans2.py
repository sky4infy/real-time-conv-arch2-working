"""
Translation module.

Two engines, routed by language pair:
  - IndicTrans2 (AI4Bharat) — used for ANY pair involving an Indic language
    (hi, mr, bn, gu, kn, ml, ur, pa, ta, te), including direct Indic<->Indic
    translation with no English pivot. This is a quality upgrade over the
    old Helsinki-NLP pivot-via-English approach for these languages, and
    the only real fix for Marathi/Bengali/etc quality issues — those
    languages had no direct MarianMT model at all before.
  - Helsinki-NLP MarianMT — kept exactly as before for English<->European
    pairs (fr, de, es, zh, ar), which were already working well.

IMPORTANT — read before deploying:
  - IndicTrans2 models are much heavier than the MarianMT pairs (200M-320M
    params per model vs MarianMT's much smaller pairs). You are ALREADY at
    16Gi RAM with OOM history on this project — loading these on top of
    your existing preloaded MarianMT pairs will very likely need MORE
    memory, not less. Test locally with `docker stats` or similar before
    assuming this fits in your current Cloud Run memory allocation.
  - `trust_remote_code=True` is required by these models' tokenizers (per
    AI4Bharat's own usage instructions) — this executes code shipped by
    the model repo, not just weights. Reasonable for AI4Bharat's official
    repo, but worth knowing this is a different trust model than MarianMT.
  - num_beams=5 (AI4Bharat's suggested default) is slower than the
    num_beams=1 greedy decoding your MarianMT pipeline uses. Lowered to 1
    below for latency parity with your existing pipeline — bump it back
    up if you want AI4Bharat's benchmarked quality and can afford the
    extra latency per turn.
  - This has NOT been run end-to-end in this environment. Test locally
    with real audio/text before deploying.
"""

from transformers import MarianMTModel, MarianTokenizer, AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import time

# ─── MARIAN MT (unchanged — European pairs) ────────────────────
LANGUAGE_PAIRS = {
    ("en", "hi"): "Helsinki-NLP/opus-mt-en-hi",
    ("hi", "en"): "Helsinki-NLP/opus-mt-hi-en",
    ("en", "ta"): "Helsinki-NLP/opus-mt-en-dra",
    ("ta", "en"): "Helsinki-NLP/opus-mt-dra-en",
    ("en", "te"): "Helsinki-NLP/opus-mt-en-dra",
    ("te", "en"): "Helsinki-NLP/opus-mt-dra-en",
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

# ─── INDICTRANS2 (new — all Indic pairs) ───────────────────────
# Every Indic language your app supports, matching src/deepgram_stt.py's
# WHISPER_ONLY set plus the Deepgram-supported Indic languages (hi, ta, te).
INDIC_LANGS = {"hi", "mr", "bn", "gu", "kn", "ml", "ur", "pa", "ta", "te"}

# Your app's simple 2-letter codes -> IndicTrans2's FLORES-style codes.
# NOTE: verify each of these against IndicTrans2's actual supported list
# before relying on it — transcribed here from AI4Bharat's documentation,
# double-check for any language you actually use in production.
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

# Distilled (smaller/faster) checkpoints — chosen over the 1B variants
# given this project's existing memory constraints on Cloud Run.
INDICTRANS_MODELS = {
    "en-indic": "ai4bharat/indictrans2-en-indic-dist-200M",
    "indic-en": "ai4bharat/indictrans2-indic-en-dist-200M",
    "indic-indic": "ai4bharat/indictrans2-indic-indic-dist-320M",
}

PIVOT_LANGUAGES = {"bn", "mr", "gu", "ur", "ja", "ta", "te", "kn", "ml"}


class Translator:
    def __init__(self):
        self._cache = {}          # MarianMT models, unchanged
        self._indic_cache = {}    # IndicTrans2 models + tokenizers + processor
        self._indic_processor = None
        print("[Translator] Ready.")

    # ── MarianMT loading (unchanged) ──────────────────────────
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

    # ── IndicTrans2 loading ────────────────────────────────────
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
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, trust_remote_code=True
        )
        model.eval()
        # Quantize the same way as MarianMT, for the same CPU-latency reasons.
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
                # Lowered from AI4Bharat's suggested 5 to 1 for latency
                # parity with the rest of this pipeline. Raise this if you
                # want their benchmarked quality and can accept slower turns.
                num_beams=1,
                num_return_sequences=1,
            )
        decoded = tokenizer.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        result = ip.postprocess_batch(decoded, lang=tgt_code)
        return result[0] if result else text

    # ── Public API (unchanged signature) ────────────────────────
    def translate(self, text: str, source_lang: str, target_lang: str) -> dict:
        if source_lang == target_lang:
            return {"translated_text": text, "source_lang": source_lang,
                     "target_lang": target_lang, "latency": 0.0, "method": "same"}
        if not text.strip():
            return {"translated_text": "", "source_lang": source_lang,
                     "target_lang": target_lang, "latency": 0.0, "method": "empty"}

        t0 = time.time()

        # Route: anything touching an Indic language (except a pure
        # non-Indic pair like es<->fr, which can't happen given this
        # app's language list, but kept for safety) goes to IndicTrans2.
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

        # ── Everything below is your original MarianMT logic, unchanged ──
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
            method = "pivot_via_english"
            print(f"[Translator] Pivoting: {source_lang} → en → {target_lang}")
            src_en_model = LANGUAGE_PAIRS.get((source_lang, "en"))
            if src_en_model:
                english_text = self._run(text, src_en_model)
            else:
                try:
                    english_text = self._run(text, f"Helsinki-NLP/opus-mt-{source_lang}-en")
                except Exception:
                    english_text = text
            en_tgt_model = LANGUAGE_PAIRS.get(("en", target_lang))
            if en_tgt_model:
                translated = self._run(english_text, en_tgt_model)
            else:
                try:
                    translated = self._run(english_text, f"Helsinki-NLP/opus-mt-en-{target_lang}")
                except Exception:
                    translated = english_text
                    method     = "pivot_partial"

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
        Now also preloads IndicTrans2 models when a pair in the list
        involves an Indic language, instead of the old MarianMT pair.
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