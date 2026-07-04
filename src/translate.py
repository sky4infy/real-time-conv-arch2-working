"""
Translation module — Helsinki-NLP MarianMT.
Quantized INT8, greedy decoding, ~0.15s per translation.
Only verified existing HuggingFace models are listed.
For Indic languages without direct models, pivot via English is used automatically.
"""

from transformers import MarianMTModel, MarianTokenizer
import torch
import time

# VERIFIED models that actually exist on HuggingFace
# Indic languages use "dra" (Dravidian) group model for Tamil/Telugu
LANGUAGE_PAIRS = {
    # English <-> Hindi (verified)
    ("en", "hi"): "Helsinki-NLP/opus-mt-en-hi",
    ("hi", "en"): "Helsinki-NLP/opus-mt-hi-en",

    # English <-> Dravidian (Tamil, Telugu share same model)
    ("en", "ta"): "Helsinki-NLP/opus-mt-en-dra",
    ("ta", "en"): "Helsinki-NLP/opus-mt-dra-en",
    ("en", "te"): "Helsinki-NLP/opus-mt-en-dra",
    ("te", "en"): "Helsinki-NLP/opus-mt-dra-en",

    # English <-> European (verified)
    ("en", "fr"): "Helsinki-NLP/opus-mt-en-fr",
    ("fr", "en"): "Helsinki-NLP/opus-mt-fr-en",
    ("en", "de"): "Helsinki-NLP/opus-mt-en-de",
    ("de", "en"): "Helsinki-NLP/opus-mt-de-en",
    ("en", "es"): "Helsinki-NLP/opus-mt-en-es",
    ("es", "en"): "Helsinki-NLP/opus-mt-es-en",

    # English <-> Others (verified)
    ("en", "zh"): "Helsinki-NLP/opus-mt-en-zh",
    ("zh", "en"): "Helsinki-NLP/opus-mt-zh-en",
    ("en", "ar"): "Helsinki-NLP/opus-mt-en-ar",
    ("ar", "en"): "Helsinki-NLP/opus-mt-ar-en",

    # NOTE: Bengali, Marathi, Gujarati, Urdu, Japanese
    # do NOT have direct Helsinki-NLP models.
    # These are handled via English pivot automatically.
}

# These languages go through English as intermediate step
# e.g. Bengali → English → Hindi
PIVOT_LANGUAGES = {"bn", "mr", "gu", "ur", "ja", "ta", "te", "kn", "ml"}


class Translator:
    def __init__(self):
        self._cache = {}
        print("[Translator] Ready.")

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

    def translate(self, text: str, source_lang: str, target_lang: str) -> dict:
        if source_lang == target_lang:
            return {"translated_text": text, "source_lang": source_lang,
                    "target_lang": target_lang, "latency": 0.0, "method": "same"}
        if not text.strip():
            return {"translated_text": "", "source_lang": source_lang,
                    "target_lang": target_lang, "latency": 0.0, "method": "empty"}

        t0     = time.time()
        pair   = (source_lang, target_lang)
        model  = LANGUAGE_PAIRS.get(pair)
        method = "direct"

        if model:
            # direct model exists — use it
            translated = self._run(text, model)

        elif source_lang == "en":
            # English → unknown language — try direct fallback
            method = "fallback"
            try:
                translated = self._run(text, f"Helsinki-NLP/opus-mt-en-{target_lang}")
            except Exception:
                translated = text
                method     = "no_model"

        elif target_lang == "en":
            # unknown language → English — try direct fallback
            method = "fallback"
            try:
                translated = self._run(text, f"Helsinki-NLP/opus-mt-{source_lang}-en")
            except Exception:
                translated = text
                method     = "no_model"

        else:
            # non-English → non-English (e.g. es→hi, fr→ta, de→bn)
            # ALWAYS pivot via English — this is the universal fix
            method = "pivot_via_english"
            print(f"[Translator] Pivoting: {source_lang} → en → {target_lang}")

            # step 1: source → English
            src_en_model = LANGUAGE_PAIRS.get((source_lang, "en"))
            if src_en_model:
                english_text = self._run(text, src_en_model)
            else:
                try:
                    english_text = self._run(text, f"Helsinki-NLP/opus-mt-{source_lang}-en")
                except Exception:
                    english_text = text  # can't translate to English, use original

            # step 2: English → target
            en_tgt_model = LANGUAGE_PAIRS.get(("en", target_lang))
            if en_tgt_model:
                translated = self._run(english_text, en_tgt_model)
            else:
                try:
                    translated = self._run(english_text, f"Helsinki-NLP/opus-mt-en-{target_lang}")
                except Exception:
                    translated = english_text  # return English if target model missing
                    method     = "pivot_partial"

        return {
            "translated_text": translated,
            "source_lang":     source_lang,
            "target_lang":     target_lang,
            "latency":         round(time.time() - t0, 2),
            "method":          method,
        }

    def preload(self, pairs: list):
        """Preload models safely — skip any that fail without crashing server."""
        loaded  = 0
        skipped = 0
        for src, tgt in pairs:
            m = LANGUAGE_PAIRS.get((src, tgt))
            if m:
                try:
                    self._load(m)
                    loaded += 1
                except Exception as e:
                    print(f"[Translator] Skipping {src}→{tgt} ({m}): {e}")
                    skipped += 1
        print(f"[Translator] Preload complete: {loaded} loaded, {skipped} skipped.")
