"""
ElevenLabs neural TTS module.
Generates natural-sounding voice audio in ~300ms.
Falls back to gTTS if ElevenLabs API key missing or limit reached.
"""

import asyncio
import os
import time
import aiohttp
import aiofiles
import tempfile
from dotenv import load_dotenv

load_dotenv()

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")

# ElevenLabs voice IDs — multilingual v2 model handles all languages
VOICE_IDS = {
    "en": "pNInz6obpgDQGcFmaJgB",   # Adam — free tier default voice
    "hi": "pNInz6obpgDQGcFmaJgB",
    "ta": "pNInz6obpgDQGcFmaJgB",
    "te": "pNInz6obpgDQGcFmaJgB",
    "bn": "pNInz6obpgDQGcFmaJgB",
    "fr": "pNInz6obpgDQGcFmaJgB",
    "de": "pNInz6obpgDQGcFmaJgB",
    "es": "pNInz6obpgDQGcFmaJgB",
    "ja": "pNInz6obpgDQGcFmaJgB",
}

ELEVENLABS_MODEL = "eleven_multilingual_v2"   # supports 29 languages


class ElevenLabsTTS:
    """
    Neural TTS using ElevenLabs multilingual v2.
    Returns audio as bytes (MP3) for direct WebSocket streaming.
    ~300ms latency for short phrases.
    """

    def __init__(self):
        if not ELEVENLABS_API_KEY:
            raise ValueError("[ElevenLabsTTS] ELEVENLABS_API_KEY not set in .env")
        self.api_key = ELEVENLABS_API_KEY
        self.base_url = "https://api.elevenlabs.io/v1"
        print("[ElevenLabsTTS] Ready.")

    async def synthesize(self, text: str, lang_code: str = "en") -> dict:
        """
        Convert text to speech audio bytes.

        Returns:
            {
                "audio_bytes": bytes,   # MP3 audio
                "latency":     0.35,
                "source":      "elevenlabs"
            }
        """
        if not text or not text.strip():
            return {"audio_bytes": None, "latency": 0, "source": "elevenlabs"}

        voice_id = VOICE_IDS.get(lang_code, VOICE_IDS["en"])
        url      = f"{self.base_url}/text-to-speech/{voice_id}"
        headers  = {
            "xi-api-key":   self.api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "text":       text,
            "model_id":   ELEVENLABS_MODEL,
            "voice_settings": {
                "stability":        0.5,
                "similarity_boost": 0.75,
                "style":            0.0,
                "use_speaker_boost": True,
            }
        }

        t0 = time.time()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as resp:
                    if resp.status == 200:
                        audio_bytes = await resp.read()
                        return {
                            "audio_bytes": audio_bytes,
                            "latency":     round(time.time() - t0, 2),
                            "source":      "elevenlabs",
                        }
                    else:
                        error = await resp.text()
                        print(f"[ElevenLabsTTS] API error {resp.status}: {error}")
                        return {"audio_bytes": None, "latency": 0, "source": "elevenlabs", "error": error}
        except Exception as e:
            print(f"[ElevenLabsTTS] Request failed: {e}")
            return {"audio_bytes": None, "latency": 0, "source": "elevenlabs", "error": str(e)}

    async def synthesize_to_file(self, text: str,
                                  lang_code: str = "en",
                                  filepath: str = None) -> dict:
        """Synthesize and save to file. Returns filepath."""
        result = await self.synthesize(text, lang_code)
        if result["audio_bytes"]:
            if not filepath:
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                    filepath = f.name
            async with aiofiles.open(filepath, "wb") as f:
                await f.write(result["audio_bytes"])
            result["filepath"] = filepath
        return result


class GTTSFallback:
    """
    gTTS fallback TTS — used when ElevenLabs is unavailable.
    Requires internet. Lower voice quality but 100% reliable.
    """

    def synthesize_to_file(self, text: str,
                            lang_code: str = "en",
                            filepath: str = None) -> dict:
        from gtts import gTTS
        import os

        t0       = time.time()
        gtts_map = {"zh": "zh-CN"}
        lang     = gtts_map.get(lang_code, lang_code)

        if not filepath:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                filepath = f.name

        try:
            tts = gTTS(text=text, lang=lang)
            tts.save(filepath)
            size = os.path.getsize(filepath)
            return {
                "filepath":  filepath,
                "latency":   round(time.time() - t0, 2),
                "source":    "gtts_fallback",
                "success":   size > 100,
            }
        except Exception as e:
            print(f"[GTTSFallback] Error: {e}")
            return {"filepath": None, "latency": 0, "source": "gtts_fallback", "success": False}


def get_tts():
    """
    Factory function — returns ElevenLabsTTS if API key is set,
    otherwise returns GTTSFallback. Called once at startup.
    """
    if ELEVENLABS_API_KEY:
        try:
            tts = ElevenLabsTTS()
            print("[TTS] Using ElevenLabs neural TTS.")
            return tts, "elevenlabs"
        except Exception as e:
            print(f"[TTS] ElevenLabs init failed: {e}. Using gTTS fallback.")
    print("[TTS] Using gTTS fallback.")
    return GTTSFallback(), "gtts"
