"""
Architecture 2 — FastAPI WebSocket Server
Production-ready, all known bugs fixed.

Fix log (see inline comments for details):
  - Fix 1: DeepgramSTT.on_utterance_end signature crash (deepgram_stt.py)
  - Fix 3: stale-connection cleanup wiping the live user (session.py + here)
  - Fix 4: Deepgram idle-timeout keepalive (deepgram_stt.py)
  - Fix 5: unified STT session handler — language_change now actually
           restarts/switches the STT engine instead of only relabeling it
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
# Replaces the old handle_deepgram_user / handle_whisper_user split.
#
# WHY: the STT engine (Deepgram vs Whisper) and, for Deepgram, the
# actual configured language are decided at connection time and
# were NEVER revisited afterwards. A "language_change" message only
# relabeled a variable used for translation/display — it never told
# the live Deepgram connection to listen in a different language,
# and it never moved a user from Deepgram to Whisper (or back) when
# they switched to/from a Whisper-only language (mr, bn, gu, kn, ml,
# ur, pa). This function fixes that: every language change is routed
# through switch_language(), which restarts whatever needs restarting.
async def handle_user_session(websocket, session, session_id, user_id, language):
    state = {
        "lang":         language,
        "mode":         None,          # "deepgram" | "whisper"
        "audio_q":      None,
        "result_q":     None,
        "lang_ref":     None,
        "stream_task":  None,
        "process_task": None,
        "audio_buffer": bytearray(),
        "last_audio_t": time.time(),
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
            state["audio_buffer"] = bytearray()
            return

        buf = bytes(state["audio_buffer"])
        state["audio_buffer"] = bytearray()

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

        # Deepgram's language is fixed for the lifetime of a connection —
        # switching languages WITHIN Deepgram still requires a full restart
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
            state["audio_buffer"] = bytearray()

        if new_mode == "deepgram":
            await start_deepgram(new_lang)
        else:
            state["audio_buffer"] = bytearray()
            state["last_audio_t"] = time.time()

        state["mode"] = new_mode
        return new_mode

    # initial engine start (no flush needed — nothing buffered yet)
    await switch_language(language, flush_first=False)

    try:
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive(), timeout=0.1)
            except asyncio.TimeoutError:
                if (state["mode"] == "whisper"
                        and state["audio_buffer"]
                        and (time.time() - state["last_audio_t"]) > SILENCE_SECS):
                    await whisper_process_buffer()
                continue

            # Starlette's low-level receive() can return a disconnect
            # message instead of raising WebSocketDisconnect — must be
            # checked explicitly or the next receive() call crashes.
            if data.get("type") == "websocket.disconnect":
                break

            if "bytes" in data and data["bytes"]:
                session.touch()
                if state["mode"] == "deepgram":
                    await state["audio_q"].put(data["bytes"])
                else:
                    state["audio_buffer"].extend(data["bytes"])
                    state["last_audio_t"] = time.time()

            elif "text" in data:
                try:
                    msg   = json.loads(data["text"])
                    mtype = msg.get("type")
                    if mtype == "ping":
                        await safe_send_json(websocket, {"type": "pong"})
                    elif mtype == "stop":
                        break
                    elif mtype == "language_change":
                        new_lang = msg.get("language", state["lang"])
                        new_mode = await switch_language(new_lang)
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
        # Fix 3: pass websocket so a stale/late cleanup can never delete
        # a fresher reconnection that already re-registered under the
        # same user_id.
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
async def translate_and_deliver(text, source_lang, session, session_id, sender_user_id):
    loop = asyncio.get_running_loop()

    for other_id in session.get_other_users(sender_user_id):
        other = session.users.get(other_id)
        if not other:
            continue

        target_lang = other.language

        result = await loop.run_in_executor(
            None, lambda t=text, s=source_lang, tl=target_lang:
                translator.translate(t, s, tl)
        )
        translated = result["translated_text"].strip()
        if not translated:
            continue

        # send translation to sender's panel ("Translated for X" box)
        sender = session.users.get(sender_user_id)
        if sender:
            await safe_send_json(sender.websocket, {
                "type":        "translation",
                "text":        translated,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "original":    text,
            })

        # also send transcript of sender's speech to recipient so the
        # OTHER user sees what was said (two-way conversation display)
        await safe_send_json(other.websocket, {
            "type":     "other_transcript",
            "text":     text,
            "language": source_lang,
        })

        # send translated audio to recipient
        audio_b64 = await synthesize_audio(translated, target_lang)
        if audio_b64:
            await safe_send_json(other.websocket, {
                "type":        "audio",
                "data":        audio_b64,
                "target_lang": target_lang,
            })


# ─── TTS ──────────────────────────────────────────────────────
async def synthesize_audio(text: str, lang_code: str):
    loop = asyncio.get_running_loop()

    if tts_type == "elevenlabs":
        try:
            result = await tts_engine.synthesize(text, lang_code)
            if result and result.get("audio_bytes"):
                return base64.b64encode(result["audio_bytes"]).decode("utf-8")
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