"""
Architecture 2 — FastAPI WebSocket Server
Production-ready.

Fix log (see inline comments for details):
  - Fix 1: DeepgramSTT.on_utterance_end signature crash (deepgram_stt.py)
  - Fix 3: stale-connection cleanup wiping the live user (session.py + here)
  - Fix 4: Deepgram idle-timeout keepalive (deepgram_stt.py)
  - Fix 5: unified STT session handler — language_change now actually
           restarts/switches the STT engine instead of only relabeling it
  - Fix 6: Whisper buffer was only ever flushed on "no bytes arrived for
           1.5s", but the frontend streams PCM continuously (silence
           included) the entire time the mic is on, so that condition
           could never trigger mid-utterance — the buffer just grew
           until language_change/disconnect, producing long, hallucinated
           Whisper output. Now flushed on actual silence, checked per
           incoming chunk.
  - Fix 6b: the first silence-detection attempt used one fixed RMS
           threshold, which doesn't hold up against getUserMedia's
           autoGainControl (it actively boosts quiet audio toward a
           target loudness, so a real pause can still read well above a
           fixed number) — this was reported as "Whisper languages only
           transcribe on mic-off, unlike Deepgram languages which
           transcribe live." Replaced with a per-connection ADAPTIVE
           noise floor (SilenceDetector) plus a hard MAX_BUFFER_SECS
           ceiling as a safety net regardless of how well the detector
           tunes to a given room/mic.
  - Fix 7: the STT engine (esp. Deepgram) was started immediately on
           websocket connect / language_change, before the frontend had
           actually started streaming mic audio (that only happens on
           the user clicking "Speak"). Deepgram would then idle-timeout
           with a 1011 before the user ever spoke. STT now only starts
           on an explicit "start_recording" message from the client,
           sent right before it opens the mic pipeline.
"""

import asyncio
import json
import os
import time
import base64
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()

from src.translate import Translator
from src.session import session_manager
from src.deepgram_stt import DeepgramSTT, WhisperFallbackSTT, use_deepgram_for, WHISPER_ONLY
from src.elevenlabs_tts import get_tts, GTTSFallback

translator    = None
stt_deepgram  = None
stt_whisper   = None
tts_engine    = None
tts_type      = None
gtts_fallback = None
USE_DEEPGRAM  = bool(os.getenv("DEEPGRAM_API_KEY", "").strip())

elevenlabs_disabled = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global translator, stt_deepgram, stt_whisper, tts_engine, tts_type, gtts_fallback

    print("=" * 60)
    print("Architecture 2 — Loading models...")
    print(f"  Deepgram:    {'YES' if USE_DEEPGRAM else 'NO'}")
    print(f"  ElevenLabs:  {'YES' if os.getenv('ELEVENLABS_API_KEY','').strip() else 'NO'}")
    print("=" * 60)

    translator = Translator()
    translator.preload([
        ("en", "hi"), ("hi", "en"),
        ("en", "ta"), ("ta", "en"),
        ("en", "te"), ("te", "en"),
        ("en", "fr"), ("fr", "en"),
        ("en", "de"), ("de", "en"),
        ("en", "es"), ("es", "en"),
        ("en", "zh"), ("zh", "en"),
        ("en", "ar"), ("ar", "en"),
    ])

    # Whisper ALWAYS loaded — used for Marathi/Bengali/Indic + fallback
    stt_whisper = WhisperFallbackSTT()
    print("[Server] WhisperSTT: ready")

    if USE_DEEPGRAM:
        try:
            stt_deepgram = DeepgramSTT()
            print("[Server] Deepgram Nova-2: ready")
        except Exception as e:
            print(f"[Server] Deepgram failed: {e}")
            stt_deepgram = None

    tts_engine, tts_type = get_tts()
    gtts_fallback = GTTSFallback()
    print(f"[Server] TTS: {tts_type}")

    asyncio.create_task(cleanup_loop())
    print("\n[Server] Ready.\n")
    yield
    print("[Server] Shutting down.")


app = FastAPI(title="Lensara Multilingual Platform", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def cleanup_loop():
    while True:
        await asyncio.sleep(300)
        session_manager.cleanup_expired()


@app.get("/health")
async def health():
    return {
        "status":          "ok",
        "stt":             "deepgram" if stt_deepgram else "whisper",
        "tts":             tts_type,
        "translation":     "helsinki-nlp",
        "active_sessions": session_manager.active_count,
        "whisper_langs":   list(WHISPER_ONLY),
    }


# FIX 2 (this round): a single fixed RMS threshold doesn't hold up
# across different mics/environments, and specifically fights against
# getUserMedia's autoGainControl: true (already set in the frontend's
# constraints), which actively boosts quiet audio toward a target
# loudness. That means the noise floor DURING a genuine pause can sit
# well above any fixed threshold picked in advance — so silence was
# essentially never detected mid-recording, and the buffer only ever
# flushed when the mic hard-stopped (recording toggled off), which is
# exactly the symptom reported: Marathi/Whisper only produced a
# transcript on mic-off, while Deepgram languages transcribed live,
# because Deepgram's VAD is a trained model, not a fixed number.
#
# Fix: track a per-connection ADAPTIVE noise floor instead of one global
# constant. A chunk counts as silent if it's within a small multiple of
# the recently-observed floor, and the floor itself only adapts during
# quiet stretches (so a sustained loud sentence doesn't drag the floor
# up and start looking "silent" to itself).
class SilenceDetector:
    def __init__(self, multiplier: float = 2.5, min_floor: float = 150.0, alpha: float = 0.05):
        self.noise_floor = None
        self.multiplier  = multiplier
        self.min_floor   = min_floor
        self.alpha       = alpha

    def is_silent(self, chunk_bytes: bytes) -> bool:
        samples = np.frombuffer(chunk_bytes, dtype=np.int16)
        if samples.size == 0:
            return True
        rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))

        if self.noise_floor is None:
            # seed the floor from the very first chunk — assume it's
            # quiet (true often enough in practice, and self-corrects
            # within a second or two either way)
            self.noise_floor = max(rms, self.min_floor)
            return True

        threshold = max(self.noise_floor * self.multiplier, self.min_floor)
        silent    = rms < threshold

        if silent:
            self.noise_floor = (1 - self.alpha) * self.noise_floor + self.alpha * rms

        return silent


# FIX: hard ceiling on buffer duration, independent of silence detection
# working well or not — Whisper gets forced to run at least this often
# so a live session always feels roughly real-time, even in a noisy
# room where the adaptive detector above stays cautious.
MAX_BUFFER_SECS = 8.0


@app.websocket("/ws/{session_id}/{user_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str, user_id: str):
    await websocket.accept()
    print(f"[WS] Connected: session={session_id} user={user_id}")

    session = session_manager.get_or_create(session_id)

    try:
        init_raw = await asyncio.wait_for(websocket.receive_text(), timeout=10.0)
        init     = json.loads(init_raw)
        language = init.get("language", "en")
    except asyncio.TimeoutError:
        await websocket.send_json({"type": "error", "message": "Init timeout"})
        await websocket.close()
        return
    except Exception as e:
        await websocket.send_json({"type": "error", "message": f"Init failed: {e}"})
        await websocket.close()
        return

    session.add_user(user_id, websocket, language)
    session.touch()

    stt_label = "deepgram" if (stt_deepgram and use_deepgram_for(language)) else "whisper"
    await websocket.send_json({
        "type": "connected", "language": language,
        "stt": stt_label, "tts": tts_type,
    })
    print(f"[WS] {user_id} lang={language} stt={stt_label}")

    await handle_user_session(websocket, session, session_id, user_id, language)


# ─── UNIFIED USER SESSION HANDLER ──────────────────────────────
async def handle_user_session(websocket, session, session_id, user_id, language):
    state = {
        "lang":         language,
        "mode":         None,          # "deepgram" | "whisper" | None (not yet started)
        "started":      False,         # Fix 7: STT engine only runs once user presses Speak
        "audio_q":      None,
        "result_q":     None,
        "lang_ref":     None,
        "stream_task":  None,
        "process_task": None,
        "audio_buffer": bytearray(),
        "last_audio_t": time.time(),
        "buffer_start_t": None,        # FIX: when the current buffer started filling, for MAX_BUFFER_SECS
        "silence_detector": SilenceDetector(),   # FIX: adaptive per-connection noise floor
    }

    SILENCE_SECS = 1.5
    MIN_BYTES    = int(16000 * 0.5 * 2)   # 0.5s minimum before Whisper runs
    whisper_lock = asyncio.Lock()          # prevents concurrent Whisper calls

    def desired_mode_for(lang: str) -> str:
        return "deepgram" if (stt_deepgram and use_deepgram_for(lang)) else "whisper"

    async def stop_deepgram():
        if state["stream_task"]:
            state["stream_task"].cancel()
        if state["process_task"]:
            state["process_task"].cancel()
        if state["audio_q"]:
            try:
                state["audio_q"].put_nowait(None)
            except Exception:
                pass
        state["stream_task"]  = None
        state["process_task"] = None
        state["audio_q"]      = None
        state["result_q"]     = None
        state["lang_ref"]     = None

    async def start_deepgram(lang: str):
        audio_q  = asyncio.Queue()
        result_q = asyncio.Queue()
        lang_ref = [lang]
        state["audio_q"]  = audio_q
        state["result_q"] = result_q
        state["lang_ref"] = lang_ref
        state["stream_task"] = asyncio.create_task(
            stt_deepgram.transcribe_stream(audio_q, result_q, lang_ref=lang_ref)
        )
        state["process_task"] = asyncio.create_task(
            process_transcripts(result_q, session, session_id, user_id, lang_ref)
        )

    async def whisper_process_buffer():
        if whisper_lock.locked():
            return
        if len(state["audio_buffer"]) < MIN_BYTES:
            state["audio_buffer"]   = bytearray()
            state["buffer_start_t"] = None
            return

        buf = bytes(state["audio_buffer"])
        state["audio_buffer"]   = bytearray()
        state["buffer_start_t"] = None

        async with whisper_lock:
            audio_np = np.frombuffer(buf, dtype=np.int16).astype(np.float32) / 32768.0
            loop         = asyncio.get_running_loop()
            current_lang = state["lang"]
            result       = await loop.run_in_executor(
                None,
                lambda: stt_whisper.transcribe(
                    audio_np,
                    language=current_lang,
                    allowed_languages=[current_lang, "en"]
                )
            )
            text = result.get("text", "").strip()
            if not text:
                return
            await safe_send_json(websocket, {
                "type":     "transcript",
                "text":     text,
                "language": result["language"],
                "is_final": True,
            })
            await translate_and_deliver(
                text, result["language"], session, session_id, user_id
            )

    async def switch_language(new_lang: str, flush_first: bool = True):
        new_mode = desired_mode_for(new_lang)
        old_mode = state["mode"]

        # flush whatever's pending under the OLD language first, so the
        # last thing said before switching isn't lost
        if flush_first and old_mode == "whisper" and state["audio_buffer"]:
            await whisper_process_buffer()

        needs_restart = (
            new_mode != old_mode or
            (new_mode == "deepgram" and new_lang != state["lang"])
        )

        state["lang"] = new_lang

        if not needs_restart:
            return new_mode

        if old_mode == "deepgram":
            await stop_deepgram()
        elif old_mode == "whisper":
            state["audio_buffer"]   = bytearray()
            state["buffer_start_t"] = None

        if new_mode == "deepgram":
            await start_deepgram(new_lang)
        else:
            state["audio_buffer"]      = bytearray()
            state["buffer_start_t"]    = None
            state["last_audio_t"]      = time.time()
            state["silence_detector"]  = SilenceDetector()   # fresh noise floor for the new language/stream

        state["mode"] = new_mode
        return new_mode

    # Fix 7: do NOT start any STT engine on connect. The frontend only
    # begins streaming audio once the user presses "Speak"; starting
    # Deepgram here means it sits open and idle until that click (or the
    # mic pipeline finishes setting up), and Deepgram closes idle
    # connections with a 1011 well before that. Engine now starts on the
    # explicit "start_recording" message below.

    try:
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive(), timeout=0.1)
            except asyncio.TimeoutError:
                if state["mode"] == "whisper" and state["audio_buffer"]:
                    silence_elapsed = time.time() - state["last_audio_t"]
                    buffer_age = (
                        time.time() - state["buffer_start_t"]
                        if state["buffer_start_t"] else 0.0
                    )
                    # FIX: flush on EITHER a genuine detected pause OR the
                    # buffer just having run long enough — whichever comes
                    # first. The duration cap is the safety net: even if
                    # the adaptive detector below is being too cautious
                    # for a given room/mic, the session still gets
                    # periodic transcripts instead of silence until the
                    # user manually stops.
                    if silence_elapsed > SILENCE_SECS or buffer_age > MAX_BUFFER_SECS:
                        await whisper_process_buffer()
                continue

            # Starlette's low-level receive() can return a disconnect
            # message instead of raising WebSocketDisconnect — must be
            # checked explicitly or the next receive() call crashes.
            if data.get("type") == "websocket.disconnect":
                break

            if "bytes" in data and data["bytes"]:
                session.touch()
                if state["mode"] == "deepgram" and state["audio_q"]:
                    await state["audio_q"].put(data["bytes"])
                elif state["mode"] == "whisper":
                    if state["buffer_start_t"] is None:
                        state["buffer_start_t"] = time.time()
                    state["audio_buffer"].extend(data["bytes"])
                    # FIX: adaptive per-session noise floor instead of a
                    # fixed RMS threshold — see SilenceDetector above for
                    # why a fixed number doesn't hold up against AGC.
                    if not state["silence_detector"].is_silent(data["bytes"]):
                        state["last_audio_t"] = time.time()
                # else: mode is None (not started yet) — drop stray bytes

            elif "text" in data:
                try:
                    msg   = json.loads(data["text"])
                    mtype = msg.get("type")

                    if mtype == "ping":
                        await safe_send_json(websocket, {"type": "pong"})

                    elif mtype == "stop":
                        break

                    elif mtype == "start_recording":
                        # Fix 7: this is the only place an STT engine
                        # actually gets started.
                        if not state["started"]:
                            new_mode = await switch_language(state["lang"], flush_first=False)
                            state["started"] = True
                            print(f"[WS] {user_id} start_recording (stt={new_mode})")

                    elif mtype == "stop_recording":
                        if state["started"]:
                            if state["mode"] == "whisper" and state["audio_buffer"]:
                                await whisper_process_buffer()
                            await stop_deepgram()
                            state["mode"]    = None
                            state["started"] = False
                            print(f"[WS] {user_id} stop_recording")

                    elif mtype == "language_change":
                        new_lang = msg.get("language", state["lang"])
                        if state["started"]:
                            # live switch — engine is running, do the
                            # full flush/restart dance
                            new_mode = await switch_language(new_lang)
                        else:
                            # nothing running yet (dropdown is disabled
                            # while recording anyway) — just remember
                            # the preference, no engine to restart
                            state["lang"] = new_lang
                            new_mode = desired_mode_for(new_lang)
                        if user_id in session.users:
                            session.users[user_id].language = new_lang
                        print(f"[WS] {user_id} language → {new_lang} (stt={new_mode})")
                        await safe_send_json(websocket, {
                            "type":     "language_updated",
                            "language": new_lang,
                            "stt":      new_mode,
                        })

                    elif mtype == "mic_restart":
                        await safe_send_json(websocket, {"type": "mic_restart_ack"})

                except Exception:
                    pass

    except WebSocketDisconnect:
        print(f"[WS] Disconnected: {user_id}")
    except Exception as e:
        print(f"[WS] Error {user_id}: {e}")
    finally:
        if state["mode"] == "whisper" and state["audio_buffer"]:
            await whisper_process_buffer()
        await stop_deepgram()

        session.remove_user(user_id, websocket)

# ─── TRANSCRIPT PROCESSOR ─────────────────────────────────────
async def process_transcripts(result_q, session, session_id, user_id, lang_ref):
    while True:
        try:
            result = await asyncio.wait_for(result_q.get(), timeout=120.0)
        except asyncio.TimeoutError:
            continue
        except asyncio.CancelledError:
            break

        if not result or "error" in result:
            continue

        text     = result.get("text", "").strip()
        is_final = result.get("is_final", False)
        lang     = result.get("language", lang_ref[0])

        if not text:
            continue

        user_state = session.users.get(user_id)
        if user_state:
            await safe_send_json(user_state.websocket, {
                "type": "transcript", "text": text,
                "language": lang, "is_final": is_final,
            })

        if is_final:
            await translate_and_deliver(text, lang, session, session_id, user_id)


# ─── TRANSLATE + DELIVER ──────────────────────────────────────
async def _deliver_to_one(other_id, text, source_lang, session, sender_user_id):
    loop  = asyncio.get_running_loop()
    other = session.users.get(other_id)
    if not other:
        return

    target_lang = other.language

    # Latency fix: send the raw transcript to the listener immediately —
    # it doesn't depend on translation finishing, so there's no reason
    # to make them wait for it. Same text as before, just sent sooner.
    await safe_send_json(other.websocket, {
        "type":     "other_transcript",
        "text":     text,
        "language": source_lang,
    })

    result = await loop.run_in_executor(
        None, lambda t=text, s=source_lang, tl=target_lang:
            translator.translate(t, s, tl)
    )
    translated = result["translated_text"].strip()
    if not translated:
        return

    sender = session.users.get(sender_user_id)

    # Latency fix: notifying the sender's "Translated for X" panel and
    # generating the TTS audio are independent once translation text
    # exists — run them concurrently instead of one after the other.
    async def notify_sender():
        if sender:
            await safe_send_json(sender.websocket, {
                "type":        "translation",
                "text":        translated,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "original":    text,
            })

    async def send_audio():
        audio_b64 = await synthesize_audio(translated, target_lang)
        if audio_b64:
            await safe_send_json(other.websocket, {
                "type":        "audio",
                "data":        audio_b64,
                "target_lang": target_lang,
            })

    await asyncio.gather(notify_sender(), send_audio())


async def translate_and_deliver(text, source_lang, session, session_id, sender_user_id):
    # Latency fix: if there are ever more than 2 participants, process
    # every listener concurrently instead of one at a time. For a 2-person
    # session this is a single call either way — zero behavior change,
    # just future-proofed.
    others = session.get_other_users(sender_user_id)
    await asyncio.gather(*[
        _deliver_to_one(other_id, text, source_lang, session, sender_user_id)
        for other_id in others
    ])


# ─── TTS ──────────────────────────────────────────────────────
async def synthesize_audio(text: str, lang_code: str):
    global elevenlabs_disabled
    loop = asyncio.get_running_loop()

    if tts_type == "elevenlabs" and not elevenlabs_disabled:
        try:
            result = await tts_engine.synthesize(text, lang_code)
            if result and result.get("audio_bytes"):
                return base64.b64encode(result["audio_bytes"]).decode("utf-8")

            err_text = str(result.get("error", "")).lower()
            if "quota" in err_text or "401" in err_text:
                elevenlabs_disabled = True
                print("[TTS] ElevenLabs quota exhausted — disabling for remainder of run, using gTTS only")
            else:
                print(f"[TTS] ElevenLabs no audio for {lang_code}, using gTTS")
        except Exception as e:
            print(f"[TTS] ElevenLabs error: {e}, using gTTS")

    # gTTS fallback
    try:
        result = await loop.run_in_executor(
            None, lambda: gtts_fallback.synthesize_to_file(text, lang_code)
        )
        fp = result.get("filepath")
        if fp and os.path.exists(fp):
            with open(fp, "rb") as f:
                audio_bytes = f.read()
            try:
                os.unlink(fp)
            except Exception:
                pass
            if len(audio_bytes) > 100:
                return base64.b64encode(audio_bytes).decode("utf-8")
    except Exception as e:
        print(f"[TTS] gTTS error: {e}")
    return None


# ─── HELPERS ──────────────────────────────────────────────────
async def safe_send_json(ws, data: dict):
    try:
        await ws.send_json(data)
    except Exception:
        pass


# ─── FRONTEND ─────────────────────────────────────────────────
frontend_dist = os.path.join(os.path.dirname(__file__), "frontend", "dist")
if os.path.exists(frontend_dist):
    app.mount("/assets", StaticFiles(directory=f"{frontend_dist}/assets"), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        return FileResponse(os.path.join(frontend_dist, "index.html"))
else:
    @app.get("/")
    async def root():
        return {"message": "Frontend not built. Run: cd frontend && npm run build"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port,
                reload=False, ws_ping_interval=20, ws_ping_timeout=20)