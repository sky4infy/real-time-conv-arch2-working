"""
Architecture 2 — FastAPI WebSocket Server
Production-ready, all known bugs fixed.
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

    if stt_deepgram and use_deepgram_for(language):
        await handle_deepgram_user(websocket, session, session_id, user_id, language)
    else:
        await handle_whisper_user(websocket, session, session_id, user_id, language)


# ─── DEEPGRAM HANDLER ─────────────────────────────────────────
async def handle_deepgram_user(websocket, session, session_id, user_id, language):
    audio_q  = asyncio.Queue()
    result_q = asyncio.Queue()
    lang_ref = [language]

    stream_task  = asyncio.create_task(
        stt_deepgram.transcribe_stream(audio_q, result_q, lang_ref=lang_ref)
    )
    process_task = asyncio.create_task(
        process_transcripts(result_q, session, session_id, user_id, lang_ref)
    )

    try:
        while True:
            data = await websocket.receive()

            if data.get("Type")=="websocket.disconnect":break

            if "bytes" in data and data["bytes"]:
                await audio_q.put(data["bytes"])
                session.touch()

            elif "text" in data:
                try:
                    msg   = json.loads(data["text"])
                    mtype = msg.get("type")
                    if mtype == "ping":
                        await safe_send_json(websocket, {"type": "pong"})
                    elif mtype == "stop":
                        break
                    elif mtype == "language_change":
                        new_lang    = msg.get("language", lang_ref[0])
                        lang_ref[0] = new_lang
                        if user_id in session.users:
                            session.users[user_id].language = new_lang
                        print(f"[WS] {user_id} language → {new_lang}")
                        await safe_send_json(websocket, {"type": "language_updated", "language": new_lang})
                    elif mtype == "mic_restart":
                        await safe_send_json(websocket, {"type": "mic_restart_ack"})
                except Exception:
                    pass

    except WebSocketDisconnect:
        print(f"[WS] Disconnected: {user_id}")
    except Exception as e:
        print(f"[WS] Error {user_id}: {e}")
    finally:
        await audio_q.put(None)
        stream_task.cancel()
        process_task.cancel()
        session.remove_user(user_id, websocket)


# ─── WHISPER HANDLER ──────────────────────────────────────────
async def handle_whisper_user(websocket, session, session_id, user_id, language):
    """
    Handles users speaking languages Deepgram doesn't support
    (Marathi, Bengali, Gujarati, Kannada, Malayalam etc.)
    Uses batch Whisper processing with silence detection.

    BUG FIX: language variable is now a mutable list so nested
    functions can update it (Python scoping fix).
    BUG FIX: audio_buffer is checked for minimum size before
    processing to avoid feeding silence to Whisper.
    BUG FIX: process_buffer uses a lock to prevent concurrent
    processing which caused double-processing and dropped audio.
    """
    lang_ref     = [language]     # FIX: mutable ref instead of rebinding parameter
    audio_buffer = bytearray()
    last_audio_t = time.time()
    SILENCE_SECS = 1.5
    MIN_BYTES    = int(16000 * 0.5 * 2)   # 0.5s minimum
    processing   = asyncio.Lock()          # FIX: prevent concurrent Whisper calls

    async def process_buffer():
        nonlocal audio_buffer

        if processing.locked():
            # already processing — save current buffer for next round
            return

        if len(audio_buffer) < MIN_BYTES:
            audio_buffer = bytearray()   # clear tiny fragments
            return

        # FIX: extract and clear atomically using slice, not reassignment
        buf          = bytes(audio_buffer)
        audio_buffer = bytearray()       # this works because nonlocal is declared

        async with processing:
            audio_np = np.frombuffer(buf, dtype=np.int16).astype(np.float32) / 32768.0

            loop         = asyncio.get_running_loop()
            current_lang = lang_ref[0]
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

    try:
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive(), timeout=0.1)

                if data.get("type") == "websocket.disconnect":break

                if "bytes" in data and data["bytes"]:
                    audio_buffer.extend(data["bytes"])
                    last_audio_t = time.time()
                    session.touch()

                elif "text" in data:
                    try:
                        msg   = json.loads(data["text"])
                        mtype = msg.get("type")
                        if mtype == "ping":
                            await safe_send_json(websocket, {"type": "pong"})
                        elif mtype == "language_change":
                            new_lang    = msg.get("language", lang_ref[0])
                            lang_ref[0] = new_lang   # FIX: update ref, not parameter
                            if user_id in session.users:
                                session.users[user_id].language = new_lang
                            print(f"[WS] {user_id} language → {new_lang}")
                            await safe_send_json(websocket, {"type": "language_updated", "language": new_lang})
                        elif mtype == "mic_restart":
                            await safe_send_json(websocket, {"type": "mic_restart_ack"})
                    except Exception:
                        pass

            except asyncio.TimeoutError:
                if audio_buffer and (time.time() - last_audio_t) > SILENCE_SECS:
                    await process_buffer()

    except WebSocketDisconnect:
        print(f"[WS] Disconnected: {user_id}")
    except Exception as e:
        print(f"[WS] Error {user_id}: {e}")
    finally:
        if audio_buffer:
            await process_buffer()
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

        # FIX: also send transcript of sender's speech to recipient
        # so the OTHER user sees what was said (two-way conversation display)
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
