# Real-Time Multilingual Conversation Platform

**Architecture 2** — a low-latency, two-way speech translation system that lets two people speak different languages and understand each other in near real time: speech in, translated speech out, per listener.

[![Live Demo](https://img.shields.io/badge/demo-live-3FB950)](https://real-time-conv-arch2-920695810067.us-central1.run.app) ![Python](https://img.shields.io/badge/python-3.12-1F6FEB) ![FastAPI](https://img.shields.io/badge/backend-FastAPI-1F6FEB) ![React](https://img.shields.io/badge/frontend-React%20%2F%20Vite-8957E5) ![License](https://img.shields.io/badge/license-see%20LICENSE-484F58)

🔗 **Live demo:** [real-time-conv-arch2-920695810067.us-central1.run.app](https://real-time-conv-arch2-920695810067.us-central1.run.app)
> First request after idle may take 20–30s (cold start — models reload on Cloud Run). Subsequent turns are fast.

---

## Table of contents

- [Screenshots](#screenshots)
- [What it does](#what-it-does)
- [Architecture](#architecture)
- [Speech-to-text routing](#speech-to-text-routing)
- [Translation](#translation)
- [Text-to-speech](#text-to-speech)
- [Tech stack](#tech-stack)
- [Key engineering problems solved](#key-engineering-problems-solved)
- [Project structure](#project-structure)
- [Running locally](#running-locally)
- [Deployment](#deployment)
- [Known limitations](#known-limitations)
- [License](#license)

---

## Screenshots

<p align="center">
  <img src="assets/Home.png" width="700"><br><br>
  <img src="assets/lang-selection.png" width="700"><br><br>
  <img src="assets/conversation.png" width="700"><br><br>
  <img src="assets/translation.png" width="700"><br><br>
  <img src="assets/translation2.png" width="700"><br><br>
  <img src="assets/translation3.png" width="700"><br><br>
  <img src="assets/DemoArch.png" width="700">
</p>

---

## What it does

Two participants (A and B) each pick their spoken language. When one speaks:

1. Speech is transcribed in real time.
2. The transcript is translated into the *other* participant's chosen language.
3. The translation is synthesized back into natural-sounding speech and played to them.
4. Either participant can switch their language mid-conversation, and the pipeline reconfigures live — no restart required.

Built to explore what a production-shaped, low-latency, multi-engine STT/translation/TTS pipeline actually requires once you get past the demo-in-a-notebook stage: connection lifecycle management, engine fallback, graceful degradation, and correctness under real multi-user conditions — not just a single happy-path demo.

---

## Architecture

```
┌──────────────┐        WebSocket (×2)        ┌───────────────────────────────┐
│  React SPA    │ ◄────────────────────────►   │         FastAPI server         │
│  (Vite/JSX)   │   binary PCM / JSON events    │                                 │
└──────────────┘                                │  per-connection state machine  │
                                                 │  ┌─────┐  ┌───────────┐        │
                                                 │  │ STT │─▶│Translation│        │
                                                 │  │Deepgram/│ MarianMT /│        │
                                                 │  │Whisper │ IndicTrans2│       │
                                                 │  └─────┘  └─────┬─────┘        │
                                                 │                 ▼              │
                                                 │           ┌───────────┐        │
                                                 │           │    TTS    │        │
                                                 │           │ ElevenLabs│        │
                                                 │           │  / gTTS   │        │
                                                 │           └───────────┘        │
                                                 └───────────────────────────────┘
```

**Single process.** FastAPI serves both the WebSocket/API layer and the built frontend static files — no separate translation microservice, no queue, no database. Session state, loaded models, and audio buffers all live in-process. This is a deliberate simplicity trade-off, and it's also why the deployment is pinned to a single instance (see [Deployment](#deployment)).

**Two independent sockets per session.** A "session" is a shared ID two browser tabs connect to (`/ws/{session_id}/{user_id}`). Each participant owns their own WebSocket, reconnect logic, and server-side state machine (STT mode, audio buffer, silence detector). The two pipelines only touch at one integration point: delivering a transcript/translation/audio payload to the *other* participant's socket.

**Concurrency.** Each connection runs its own `asyncio` tasks — one streaming audio to Deepgram and pulling transcripts, another processing finished transcripts into translation + delivery. Whisper runs synchronously in a thread-pool executor when a buffer is flushed, since it isn't a streaming API. Within one translation turn, independent steps (relaying the raw transcript to the listener, notifying the sender's UI, generating TTS audio) run concurrently via `asyncio.gather` rather than sequentially.

---

## Speech-to-text routing

| | |
|---|---|
| **Deepgram Nova-2 (streaming)** | English, Hindi, Tamil, Telugu, French, German, Spanish, Chinese, Arabic, and more |
| **`faster-whisper` (local, INT8, CPU)** | Marathi, Bengali, Gujarati, Kannada, Malayalam, Urdu, Punjabi — languages Deepgram doesn't support — and as a general fallback |

Routing is decided per-language and re-decided live if a participant switches languages mid-session — a `language_change` message tears down and restarts whatever engine configuration the new language requires, flushing any audio buffered under the *old* language first so nothing spoken right before a switch is lost.

The STT engine is only started on an explicit "start recording" signal from the client, not on connect — starting it eagerly on connect left Deepgram sockets idling long enough to hit Deepgram's own idle-disconnect timer before anyone had spoken.

---

## Translation

Translation has gone through three iterations, kept in the repo (`translate_prev.py`, `translateV2.py`, `translate.py`) as a visible record of why the routing changed:

**v1 — Helsinki-NLP MarianMT only.** Every pair routed through MarianMT (quantized INT8), with direct bilingual models where they exist and an English pivot elsewhere. Languages without a dedicated Helsinki model (Tamil, Telugu, Bengali, Gujarati, Kannada, Marathi) used Helsinki's multilingual group models with an explicit language tag prepended to the source text.

**v2 — introduced AI4Bharat IndicTrans2.** The multilingual Helsinki models had real, documented quality problems specifically for Indic languages: known Indic→English quality issues in the `mul-en` direction, training data for several pairs drawn largely from religious-text corpora (poor fit for conversational speech), and INT8 quantization disproportionately hurting low-resource language quality. v2 replaced every Indic-involving pair with IndicTrans2 (distilled 200M checkpoints, chosen for memory headroom on Cloud Run), leaving the already-solid European/Chinese/Arabic pairs on MarianMT.

**v3 (current) — restores correct pivoting.** v2's rewrite silently dropped English-pivot fallback for non-Indic pairs without a hand-listed model, and threw (and swallowed) a `KeyError` for pairs mixing Indic and non-Indic-non-English languages — both cases returned the *original, untranslated* text as if translation had succeeded. v3 fixes both by routing each leg of a pivot through the correct engine (non-Indic leg → MarianMT, Indic leg → IndicTrans2), checks the Hugging Face Hub directly for whether a direct model exists for a pair instead of trusting only a hand-maintained list, composes Indic↔Indic through English via the two directional IndicTrans2 checkpoints rather than the weaker dedicated `indic-indic` checkpoint, and flags suspicious output (empty or unchanged text) in the returned `method` field instead of reporting silent success.

---

## Text-to-speech

ElevenLabs multilingual neural voices (`eleven_multilingual_v2`) by default, with automatic fallback to gTTS if the API key is missing, the request fails, or the account is rate-limited/out of quota. Once a quota failure is confirmed, the server stops retrying ElevenLabs for the rest of that run instead of eating the failure latency on every subsequent turn — the conversation never breaks, it just gets a flatter voice until ElevenLabs is available again.

---

## Tech stack

| Layer | Technology |
|---|---|
| Frontend | React (Vite), vanilla WebSocket + Web Audio API |
| Backend | FastAPI, Python 3.12, asyncio |
| Speech-to-text | Deepgram Nova-2 (streaming) + faster-whisper (local fallback) |
| Translation | Helsinki-NLP MarianMT + AI4Bharat IndicTrans2 (Transformers + PyTorch, quantized) |
| Text-to-speech | ElevenLabs (neural) + gTTS (fallback) |
| Deployment | Docker, Google Cloud Run |

---

## Key engineering problems solved

This project went through real debugging rounds against actual multi-user sessions, not just single-user demos.

- **Live STT engine switching.** Deepgram's language is fixed for the lifetime of a connection, and some languages only work on Whisper at all. A `language_change` message correctly tears down and restarts whatever engine configuration the new language requires — including flushing any audio buffered under the *old* language first.
- **Stale-connection race conditions.** A reconnecting client could have its old connection's cleanup code delete a fresher, already-reconnected session entry purely by user ID. Session removal now checks connection identity, not just the key.
- **Idle-timeout disconnects.** Deepgram's connection idle timer wasn't being satisfied by naive "silence" padding; it now sends the protocol's actual documented keepalive message.
- **STT engine started too early.** Engines were starting on connect, well before the mic button was pressed, and idling out before anyone spoke. They now start only on an explicit "start recording" signal.
- **Whisper buffer never flushed mid-recording.** The original flush condition ("no audio arrived recently") could never fire, because the client streams PCM continuously the entire time the mic is on — silence included. Fixed by detecting actual silence in the audio itself, on every incoming chunk.
- **Fixed silence threshold vs. browser auto-gain control.** A fixed volume threshold doesn't hold up against the browser's automatic gain control, which actively boosts quiet audio — a genuine pause could still read as "loud." Replaced with a per-connection adaptive noise floor, plus a hard maximum-buffer-duration ceiling as a safety net.
- **Translation passthrough bugs.** Two silent-failure cases — a dropped English-pivot fallback, and a swallowed `KeyError` on Indic/non-Indic mixed pairs — both returned untranslated text as if translation had succeeded. Fixed with correct per-leg routing and explicit suspicious-output flagging.
- **Turn-latency reduction.** Independent steps in the translate → notify → synthesize pipeline (relaying the raw transcript, notifying the sender, generating audio) run concurrently instead of sequentially, and a confirmed ElevenLabs quota failure is remembered for the run instead of retried every turn.

---

## Project structure

```
.
├── server.py               # FastAPI app, WebSocket handling, pipeline orchestration
├── src/
│   ├── session.py           # In-memory session/connection management
│   ├── deepgram_stt.py      # Deepgram streaming STT + Whisper fallback
│   ├── translate.py         # Current translation module (v3 — IndicTrans2 + corrected pivoting)
│   ├── translateV2.py       # v2 — introduced IndicTrans2, kept for reference
│   ├── translate_prev.py    # v1 — MarianMT-only, kept for reference
│   └── elevenlabs_tts.py    # ElevenLabs TTS + gTTS fallback
├── frontend/
│   └── src/
│       ├── App.jsx                 # Top-level state, dual WebSocket management
│       ├── App.css                 # Dark dashboard theme
│       ├── main.jsx                # Vite/React entry point
│       └── TranslationPanel.jsx    # Per-participant mic capture + playback
├── Dockerfile
└── requirements.txt
```

---

## Running locally

### Prerequisites
- Python 3.12
- Node.js 20+
- All API keys are optional — the app runs fully offline-capable, falling back to local Whisper + gTTS:
  - `DEEPGRAM_API_KEY` — streaming STT for Deepgram-supported languages
  - `ELEVENLABS_API_KEY` — neural TTS
  - `HF_TOKEN` — optional; avoids Hugging Face Hub rate limits on the model-existence checks `translate.py` makes at runtime

### Backend

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt --break-system-packages
cp .env.example .env            # add your API keys if you have them
python server.py
```

> **Note:** `IndicTransToolkit` compiles a native extension on install. On Windows this requires Microsoft Visual C++ Build Tools (the "Desktop development with C++" workload) on `PATH` before running `pip install` — verify with `where cl.exe` in the same terminal first.
>
> `requirements.txt` pins `transformers==4.44.2`, `tokenizers==0.19.1`, and `huggingface_hub==0.26.5` deliberately — newer versions of these (commented out in the file as a reminder) break `IndicTransToolkit` compatibility. Don't upgrade them independently.

### Frontend

```bash
cd frontend
npm install
npm run build
```

The FastAPI server serves the built frontend directly — visit `http://localhost:8000`.

### Health check

```bash
curl http://localhost:8000/health
```

---

## Deployment

Deployed on **Google Cloud Run** via a multi-stage Docker build (frontend build → Python runtime). See `Dockerfile`.

```bash
gcloud run deploy real-time-conv-arch2 \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8080 \
  --memory 16Gi \
  --cpu 4 \
  --timeout 3600 \
  --session-affinity \
  --set-env-vars DEEPGRAM_API_KEY=...,ELEVENLABS_API_KEY=...,HF_TOKEN=...
```

> Session state currently lives in-process (in-memory), so this is deployed as a single instance (`--max-instances 1`). `--session-affinity` and the long `--timeout` reflect that this is a stateful, long-lived-connection workload rather than a stateless API. A multi-instance production version would move session state to Redis/Memorystore.

---

## Known limitations

- Single-session-at-a-time deployment (in-memory session store, see [Deployment](#deployment)).
- ElevenLabs quota exhaustion falls back to gTTS, which is lower voice quality.
- Whisper transcription accuracy for some Indic languages can vary depending on audio quality — an active area of tuning.
- The non-Indic ↔ Indic pivot paths restored in v3 (e.g. French→Hindi, Spanish→Tamil) had not yet been run end-to-end in a live environment as of the last update — worth verifying before treating as fully confirmed.

---

## License

See [`LICENSE`](./LICENSE).