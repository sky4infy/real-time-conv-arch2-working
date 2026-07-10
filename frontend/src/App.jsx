import { useState, useEffect, useRef, useCallback } from "react";
import TranslationPanel from "./TranslationPanel";
import "./App.css";

// const BACKEND_HTTP = `http://${window.location.hostname}:8000`;
// const BACKEND_WS   = `ws://${window.location.hostname}:8000`;

const isSecure     = window.location.protocol === "https:";
const BACKEND_HTTP = isSecure
  ? `https://${window.location.hostname}`
  : `http://${window.location.hostname}:8000`;
const BACKEND_WS   = isSecure
  ? `wss://${window.location.hostname}`
  : `ws://${window.location.hostname}:8000`;

function makeSessionId() {
  return Math.random().toString(36).substring(2, 8);
}

export default function App() {
  const [sessionId]   = useState(makeSessionId);
  const [status,      setStatus]      = useState("idle");
  const [serverInfo,  setServerInfo]  = useState(null);
  const [serverError, setServerError] = useState(false);

  // A state
  const [aLang,        setALang]        = useState("en");
  const [aTranscript,  setATranscript]  = useState("");
  const [aTranslation, setATranslation] = useState("");  // what A said, translated for B
  const [aAudio,       setAAudio]       = useState(null);
  const [aConnected,   setAConnected]   = useState(false);

  // B state
  const [bLang,        setBLang]        = useState("hi");
  const [bTranscript,  setBTranscript]  = useState("");
  const [bTranslation, setBTranslation] = useState("");  // what B said, translated for A
  const [bAudio,       setBAudio]       = useState(null);
  const [bConnected,   setBConnected]   = useState(false);

  const wsA = useRef(null);
  const wsB = useRef(null);

  const LANGUAGES = [
    { code: "en", label: "English" },
    { code: "hi", label: "Hindi" },
    { code: "ta", label: "Tamil" },
    { code: "te", label: "Telugu" },
    { code: "mr", label: "Marathi" },
    { code: "fr", label: "French" },
    { code: "de", label: "German" },
    { code: "es", label: "Spanish" },
    { code: "bn", label: "Bengali" },
    { code: "gu", label: "Gujarati" },
    { code: "kn", label: "Kannada" },
    { code: "ml", label: "Malayalam" },
    { code: "ur", label: "Urdu" },
    { code: "pa", label: "Punjabi" },
    { code: "zh", label: "Chinese" },
    { code: "ar", label: "Arabic" },
  ];

  useEffect(() => {
    fetch(`${BACKEND_HTTP}/health`)
      .then(r => r.json())
      .then(d => { setServerInfo(d); setServerError(false); })
      .catch(() => setServerError(true));
  }, []);

  function b64ToBlob(b64, mime) {
    const bin   = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return new Blob([bytes], { type: mime });
  }

  function playAudio(b64) {
    const blob = b64ToBlob(b64, "audio/mp3");
    const url  = URL.createObjectURL(blob);
    const p    = new Audio(url);
    p.play().catch(e => console.warn("Autoplay:", e));
    return url;
  }

  function createWebSocket(userId, language, handlers) {
    const ws = new WebSocket(`${BACKEND_WS}/ws/${sessionId}/${userId}`);
    ws.onopen    = () => ws.send(JSON.stringify({ type: "init", language }));
    ws.onmessage = (e) => { try { handlers(JSON.parse(e.data)); } catch(ex) {} };
    ws.onerror   = (e) => console.error(`[WS ${userId}]`, e);
    ws.onclose   = (e) => {
      console.log(`[WS ${userId}] closed ${e.code}`);
      if (userId === "user_a") setAConnected(false);
      if (userId === "user_b") setBConnected(false);
    };
    return ws;
  }

  const connectA = useCallback(() => {
    if (wsA.current) { wsA.current.close(); wsA.current = null; }

    wsA.current = createWebSocket("user_a", aLang, (msg) => {
      switch (msg.type) {
        case "connected":
          setAConnected(true);
          break;
        case "transcript":
          setATranscript(msg.text);
          break;
        case "translation":
          setATranslation(msg.text);
          break;
        case "other_transcript":
          // B spoke — show what B said on B's transcript panel
          setBTranscript(msg.text);
          break;
        case "audio":
          setAAudio(playAudio(msg.data));
          break;
        case "error":
          console.error("[A]", msg.message);
          break;
      }
    });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [aLang, sessionId]);

  const connectB = useCallback(() => {
    if (wsB.current) { wsB.current.close(); wsB.current = null; }

    wsB.current = createWebSocket("user_b", bLang, (msg) => {
      switch (msg.type) {
        case "connected":
          setBConnected(true);
          break;
        case "transcript":
          setBTranscript(msg.text);
          break;
        case "translation":
          setBTranslation(msg.text);
          break;
        case "other_transcript":
          // A spoke — show what A said on A's transcript panel
          setATranscript(msg.text);
          break;
        case "audio":
          setBAudio(playAudio(msg.data));
          break;
        case "error":
          console.error("[B]", msg.message);
          break;
      }
    });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bLang, sessionId]);

  function connectBoth() {
    connectA();
    connectB();
    setStatus("connected");
  }

  function disconnectBoth() {
    if (wsA.current) { wsA.current.close(); wsA.current = null; }
    if (wsB.current) { wsB.current.close(); wsB.current = null; }
    setAConnected(false); setBConnected(false);
    setATranscript(""); setATranslation(""); setAAudio(null);
    setBTranscript(""); setBTranslation(""); setBAudio(null);
    setStatus("idle");
  }

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-inner">
          <div className="logo">
            <span className="logo-l">L</span>
            <span className="logo-text">ensara</span>
          </div>
          <div className="header-title">
            <h1>Real-Time Multilingual Platform</h1>
            <p className="header-sub">Architecture 2 — Production Demo</p>
          </div>
          <div className="header-badges">
            {serverError && (
              <span className="badge badge-red">Backend offline</span>
            )}
            {serverInfo && !serverError && (
              <>
                <span className="badge badge-green">STT: {serverInfo.stt}</span>
                <span className="badge badge-purple">TTS: {serverInfo.tts}</span>
              </>
            )}
            <span className="badge badge-blue">Session: {sessionId}</span>
          </div>
        </div>
      </header>

      <div className="session-controls">
        {status === "idle" ? (
          <button className="btn-primary btn-large"
            onClick={connectBoth} disabled={serverError}>
            ▶ Start Session
          </button>
        ) : (
          <button className="btn-danger btn-large" onClick={disconnectBoth}>
            ■ End Session
          </button>
        )}
        <p className="session-hint">
          {serverError
            ? "Backend offline — is server.py running?"
            : status === "idle"
              ? "Select languages then click Start Session"
              : "Live — speak into mic, translation plays automatically"}
        </p>
      </div>

      <div className="panels">
        <TranslationPanel
          userId="A"
          label="Participant A"
          language={aLang}
          setLanguage={setALang}
          languages={LANGUAGES}
          websocket={wsA}
          connected={aConnected}
          sessionActive={status === "connected"}
          transcript={aTranscript}
          translation={aTranslation}
          audioUrl={aAudio}
          targetLabel="B"
        />
        <div className="panel-divider">
          <div className="divider-icon">⇄</div>
        </div>
        <TranslationPanel
          userId="B"
          label="Participant B"
          language={bLang}
          setLanguage={setBLang}
          languages={LANGUAGES}
          websocket={wsB}
          connected={bConnected}
          sessionActive={status === "connected"}
          transcript={bTranscript}
          translation={bTranslation}
          audioUrl={bAudio}
          targetLabel="A"
        />
      </div>

      <footer className="app-footer">
        <span>Lensara Technologies · Architecture 2 · Zero cost demo</span>
        <span>Deepgram → Helsinki-NLP → ElevenLabs · &lt;1.5s latency</span>
      </footer>
    </div>
  );
}
