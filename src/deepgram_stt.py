"""
Deepgram real-time streaming STT + WhisperFallbackSTT.

Key design decisions:
- Languages Deepgram supports natively: en, hi, ta, te, fr, de, es, zh, ar
- Languages Deepgram does NOT support: mr, bn, gu, kn, ml, ur
- For unsupported languages: route audio through WhisperFallbackSTT instead
- Auto-reconnect if stream drops
- 90s idle timeout before reconnect (long pauses are normal in conversation)
"""

import asyncio
import os
import numpy as np
import time
from dotenv import load_dotenv

load_dotenv()

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
SAMPLE_RATE      = 16000

# Languages Deepgram Nova-2 reliably supports
DEEPGRAM_NATIVE = {
    "en", "hi", "ta", "te", "fr", "de", "es",
    "zh", "ar", "ja", "ko", "pt", "ru", "nl",
    "it", "tr", "id", "sv", "pl", "uk",
}

# Languages that must use Whisper (Deepgram does not support them)
WHISPER_ONLY = {"mr", "bn", "gu", "kn", "ml", "ur", "pa"}


def use_deepgram_for(lang_code: str) -> bool:
    """Return True if Deepgram should be used for this language."""
    return lang_code in DEEPGRAM_NATIVE


class DeepgramSTT:
    def __init__(self):
        if not DEEPGRAM_API_KEY or "your_" in DEEPGRAM_API_KEY:
            raise ValueError("[DeepgramSTT] DEEPGRAM_API_KEY not set in .env")
        from deepgram import DeepgramClient
        self.client = DeepgramClient(DEEPGRAM_API_KEY)
        print("[DeepgramSTT] Deepgram client ready.")

    async def transcribe_stream(self,
                                audio_queue: asyncio.Queue,
                                result_queue: asyncio.Queue,
                                language: str = None,
                                lang_ref: list = None):
        """
        Stream audio to Deepgram with auto-reconnect.
        lang_ref: mutable [lang_code] list — update mid-stream for language changes.
        """
        if lang_ref is None:
            lang_ref = [language or "en"]

        while True:
            should_reconnect = await self._run_stream(audio_queue, result_queue, lang_ref)
            if not should_reconnect:
                break
            print("[DeepgramSTT] Reconnecting in 0.5s...")
            await asyncio.sleep(0.5)

    async def _run_stream(self, audio_queue, result_queue, lang_ref):
        from deepgram import LiveTranscriptionEvents, LiveOptions

        lang     = lang_ref[0] if lang_ref[0] in DEEPGRAM_NATIVE else "en"
        interim  = [""]

        options = LiveOptions(
            model="nova-2",
            language=lang,
            smart_format=True,
            encoding="linear16",
            channels=1,
            sample_rate=SAMPLE_RATE,
            interim_results=True,
            utterance_end_ms="1000",
            vad_events=True,
            endpointing="300",
        )

        conn = self.client.listen.asynclive.v("1")

        async def on_transcript(_, result, **kw):
            alt      = result.channel.alternatives[0]
            sentence = alt.transcript.strip()
            if not sentence:
                return
            is_final = result.is_final
            if not is_final:
                interim[0] = sentence
            else:
                interim[0] = ""
            await result_queue.put({
                "text":     sentence,
                "is_final": is_final,
                "language": lang_ref[0],
                "source":   "deepgram",
            })

        async def on_utterance_end(self, *args, **kwargs):
            # UtteranceEnd events don't carry a `result` transcript object
            # like Transcript events do — don't rely on positional args here.
            if interim[0]:
                await result_queue.put({
                    "text":     interim[0],
                    "is_final": True,
                    "language": lang_ref[0],
                    "source":   "deepgram_ue",
                })
                interim[0] = ""

        async def on_error(_, error, **kw):
            print(f"[DeepgramSTT] Error: {error}")

        conn.on(LiveTranscriptionEvents.Transcript,   on_transcript)
        conn.on(LiveTranscriptionEvents.UtteranceEnd, on_utterance_end)
        conn.on(LiveTranscriptionEvents.Error,        on_error)

        started = await conn.start(options)
        if not started:
            print("[DeepgramSTT] Failed to start — will retry")
            return True

        print(f"[DeepgramSTT] Stream started (lang={lang})")
        should_reconnect = False

        while True:
            try:
                chunk = await asyncio.wait_for(audio_queue.get(), timeout=8.0)
                if chunk is None:
                    should_reconnect = False
                    break
                await conn.send(chunk)
            except asyncio.TimeoutError:
                # short idle — keep connection alive with keepalive
                try:
                    await conn.send(bytes(320))  # send silence to keep connection open
                except Exception:
                    should_reconnect = True
                    break
            except Exception as e:
                print(f"[DeepgramSTT] Send error: {e}")
                should_reconnect = True
                break

        try:
            await conn.finish()
        except Exception:
            pass

        return should_reconnect


class WhisperFallbackSTT:
    """
    Whisper-based STT for languages Deepgram doesn't support
    (Marathi, Bengali, Gujarati, Kannada, Malayalam, Punjabi)
    AND as a fallback when Deepgram is unavailable.
    """

    def __init__(self):
        from faster_whisper import WhisperModel
        print("[WhisperSTT] Loading faster-whisper small (better Indic accuracy)...")
        self.model = WhisperModel(
            "small", device="cpu", compute_type="int8",
            download_root="models/whisper"
        )
        # warmup
        dummy = np.zeros(16000, dtype=np.float32)
        list(self.model.transcribe(dummy, language="en", beam_size=1)[0])
        print("[WhisperSTT] Ready.")

    def transcribe(self, audio: np.ndarray,
                   language: str = None,
                   allowed_languages: list = None) -> dict:
        """Transcribe a complete audio segment."""
        audio = audio.astype(np.float32)
        m     = np.abs(audio).max()
        if m > 1.0:
            audio = audio / 32768.0
            m     = np.abs(audio).max()
        if m > 0:
            audio = audio / m * 0.9

        t0 = time.time()

        # force language if specified — critical for Marathi/Bengali etc.
        segments, info = self.model.transcribe(
            audio,
            language=language,       # forced language prevents misdetection
            beam_size=2,             # slightly better than 1 for Indic
            best_of=2,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=400, speech_pad_ms=100),
            condition_on_previous_text=False,
            temperature=0.0,
        )
        text = " ".join(s.text.strip() for s in segments).strip()

        # low confidence fallback only if language was NOT forced
        if language is None and info.language_probability < 0.6:
            segments, info = self.model.transcribe(
                audio, language="en",
                beam_size=1, best_of=1, vad_filter=True,
                condition_on_previous_text=False, temperature=0.0,
            )
            text = " ".join(s.text.strip() for s in segments).strip()

        detected = language if language else info.language
        if allowed_languages and detected not in allowed_languages:
            detected = allowed_languages[0]

        return {
            "text":     text,
            "language": detected,
            "latency":  round(time.time() - t0, 2),
            "source":   "whisper",
        }
