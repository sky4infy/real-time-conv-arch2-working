// import { useState, useEffect, useRef, useCallback } from "react";
// import TranslationPanel from "./TranslationPanel";
// import "./App.css";

// // FIXED: use port 8000 explicitly, always point to backend
// const BACKEND_HTTP = `http://${window.location.hostname}:8000`;
// const BACKEND_WS   = `ws://${window.location.hostname}:8000`;

// function makeSessionId() {
//   return Math.random().toString(36).substring(2, 8);
// }

// export default function App() {
//   const [sessionId]   = useState(makeSessionId);
//   const [status, setStatus] = useState("idle");
//   const [serverInfo, setServerInfo] = useState(null);
//   const [serverError, setServerError] = useState(false);

//   const [aLang, setALang]               = useState("en");
//   const [aTranscript, setATranscript]   = useState("");
//   const [aTranslation, setATranslation] = useState("");
//   const [aAudio, setAAudio]             = useState(null);
//   const [aLatency, setALatency]         = useState(null);
//   const [aConnected, setAConnected]     = useState(false);

//   const [bLang, setBLang]               = useState("hi");
//   const [bTranscript, setBTranscript]   = useState("");
//   const [bTranslation, setBTranslation] = useState("");
//   const [bAudio, setBAudio]             = useState(null);
//   const [bLatency, setBLatency]         = useState(null);
//   const [bConnected, setBConnected]     = useState(false);

//   const wsA = useRef(null);
//   const wsB = useRef(null);

//   const LANGUAGES = [
//     { code: "en", label: "English" },
//     { code: "hi", label: "Hindi" },
//     { code: "ml", label: "Malayalam" },
//     { code: "te", label: "Telugu" },
//     { code: "mr", label: "Marathi" },
//     { code: "fr", label: "French" },
//     { code: "de", label: "German" },
//     { code: "es", label: "Spanish" },
//     { code: "zh", label: "Chinese" },
//     { code: "ar", label: "Arabic" },
//   ];

//   // health check on load
//   useEffect(() => {
//     fetch(`${BACKEND_HTTP}/health`)
//       .then(r => r.json())
//       .then(data => { setServerInfo(data); setServerError(false); })
//       .catch(() => setServerError(true));
//   }, []);

//   // FIXED: b64ToBlob moved outside callbacks so it's stable
//   function b64ToBlob(b64, mimeType) {
//     const binary = atob(b64);
//     const bytes  = new Uint8Array(binary.length);
//     for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
//     return new Blob([bytes], { type: mimeType });
//   }

//   function createWebSocket(userId, language, onMessage, onConnect) {
//     const url = `${BACKEND_WS}/ws/${sessionId}/${userId}`;
//     const ws  = new WebSocket(url);

//     ws.onopen = () => {
//       // FIXED: send init immediately on open — don't wait
//       ws.send(JSON.stringify({ type: "init", language }));
//     };

//     ws.onmessage = (evt) => {
//       try {
//         const msg = JSON.parse(evt.data);
//         onMessage(msg);
//       } catch (e) {
//         console.error("WS parse error:", e);
//       }
//     };

//     ws.onerror = (e) => {
//       console.error(`[WS ${userId}] error`, e);
//     };

//     ws.onclose = (e) => {
//       console.log(`[WS ${userId}] closed code=${e.code}`);
//       // clear connected state
//       if (userId === "user_a") setAConnected(false);
//       if (userId === "user_b") setBConnected(false);
//     };

//     return ws;
//   }

//   const connectA = useCallback(() => {
//     if (wsA.current) { wsA.current.close(); wsA.current = null; }
//     let utteranceStart = null;

//     wsA.current = createWebSocket("user_a", aLang, (msg) => {
//       if (msg.type === "connected")   { setAConnected(true); }
//       if (msg.type === "transcript" && msg.is_final) {
//         setATranscript(msg.text);
//         utteranceStart = Date.now();   // start timer when final transcript lands
//       }
//       if (msg.type === "transcript" && !msg.is_final) {
//         setATranscript(msg.text);
//       }
//       if (msg.type === "translation") {
//         setATranslation(msg.text);
//       }
//       if (msg.type === "audio") {
//         const blob = b64ToBlob(msg.data, "audio/mp3");
//         const url  = URL.createObjectURL(blob);
//         setAAudio(url);
//         const player = new Audio(url);
//         player.play().catch(e => console.warn("[Audio A] Autoplay blocked, click page once:", e));
//         if (utteranceStart) {
//           setALatency(((Date.now() - utteranceStart) / 1000).toFixed(2));
//           utteranceStart = null;
//         }
//       }
//       if (msg.type === "error") {
//         console.error("[WS user_a] server error:", msg.message);
//       }
//     });
//   // eslint-disable-next-line react-hooks/exhaustive-deps
//   }, [aLang, sessionId]);

//   const connectB = useCallback(() => {
//     if (wsB.current) { wsB.current.close(); wsB.current = null; }
//     let utteranceStart = null;

//     wsB.current = createWebSocket("user_b", bLang, (msg) => {
//       if (msg.type === "connected")   { setBConnected(true); }
//       if (msg.type === "transcript" && msg.is_final) {
//         setBTranscript(msg.text);
//         utteranceStart = Date.now();
//       }
//       if (msg.type === "transcript" && !msg.is_final) {
//         setBTranscript(msg.text);
//       }
//       if (msg.type === "translation") {
//         setBTranslation(msg.text);
//       }
//       if (msg.type === "audio") {
//         const blob = b64ToBlob(msg.data, "audio/mp3");
//         const url  = URL.createObjectURL(blob);
//         setBAudio(url);
//         const player = new Audio(url);
//         player.play().catch(e => console.warn("[Audio B] Autoplay blocked, click page once:", e));
//         if (utteranceStart) {
//           setBLatency(((Date.now() - utteranceStart) / 1000).toFixed(2));
//           utteranceStart = null;
//         }
//       }
//       if (msg.type === "error") {
//         console.error("[WS user_b] server error:", msg.message);
//       }
//     });
//   // eslint-disable-next-line react-hooks/exhaustive-deps
//   }, [bLang, sessionId]);

//   function connectBoth() {
//     connectA();
//     connectB();
//     setStatus("connected");
//   }

//   function disconnectBoth() {
//     if (wsA.current) { wsA.current.close(); wsA.current = null; }
//     if (wsB.current) { wsB.current.close(); wsB.current = null; }
//     setAConnected(false);
//     setBConnected(false);
//     setATranscript(""); setATranslation(""); setAAudio(null);
//     setBTranscript(""); setBTranslation(""); setBAudio(null);
//     setStatus("idle");
//   }

//   return (
//     <div className="app">
//       <header className="app-header">
//         <div className="header-inner">
//           <div className="logo">
//             <span className="logo-l">L</span>
//             <span className="logo-text">ensara</span>
//           </div>
//           <div className="header-title">
//             <h1>Real-Time Multilingual Platform</h1>
//             <p className="header-sub">Architecture 2 — Production Demo</p>
//           </div>
//           <div className="header-badges">
//             {serverError && (
//               <span className="badge badge-red">Backend offline — start server.py</span>
//             )}
//             {serverInfo && !serverError && (
//               <>
//                 <span className="badge badge-green">STT: {serverInfo.stt}</span>
//                 <span className="badge badge-purple">TTS: {serverInfo.tts}</span>
//               </>
//             )}
//             <span className="badge badge-blue">Session: {sessionId}</span>
//           </div>
//         </div>
//       </header>

//       <div className="session-controls">
//         {status === "idle" ? (
//           <button
//             className="btn-primary btn-large"
//             onClick={connectBoth}
//             disabled={serverError}
//           >
//             ▶ Start Session
//           </button>
//         ) : (
//           <button className="btn-danger btn-large" onClick={disconnectBoth}>
//             ■ End Session
//           </button>
//         )}
//         <p className="session-hint">
//           {serverError
//             ? "Cannot connect — is server.py running on port 8000?"
//             : status === "idle"
//               ? "Select languages then click Start Session"
//               : "Live — speak into mic, translation plays automatically"}
//         </p>
//       </div>

//       <div className="panels">
//         <TranslationPanel
//           userId="A"
//           label="Participant A"
//           language={aLang}
//           setLanguage={setALang}
//           languages={LANGUAGES}
//           websocket={wsA}
//           connected={aConnected}
//           sessionActive={status === "connected"}
//           transcript={aTranscript}
//           translation={aTranslation}
//           audioUrl={aAudio}
//           targetLabel="B"
//         />

//         <div className="panel-divider">
//           <div className="divider-icon">⇄</div>
//         </div>

//         <TranslationPanel
//           userId="B"
//           label="Participant B"
//           language={bLang}
//           setLanguage={setBLang}
//           languages={LANGUAGES}
//           websocket={wsB}
//           connected={bConnected}
//           sessionActive={status === "connected"}
//           transcript={bTranscript}
//           translation={bTranslation}
//           audioUrl={bAudio}
//           targetLabel="A"
//         />
//       </div>

//       <footer className="app-footer">
//         <span>Lensara Technologies · Architecture 2 · Zero cost demo</span>
//         <span>Deepgram → Helsinki-NLP → ElevenLabs · &lt;1.5s latency</span>
//       </footer>
//     </div>
//   );
// }


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
