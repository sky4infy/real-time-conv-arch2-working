import { useState, useRef, useEffect, useCallback } from "react";

const SAMPLE_RATE   = 16000;
const CHUNK_SAMPLES = 2048;   // must be power of 2

export default function TranslationPanel({
  userId, label, language, setLanguage, languages,
  websocket, connected, sessionActive,
  transcript, translation, audioUrl, targetLabel
}) {
  const [recording, setRecording] = useState(false);
  const [micError,  setMicError]  = useState(null);
  const [volume,    setVolume]    = useState(0);

  // refs for audio pipeline — all mutable, no stale closure issues
  const mediaStreamRef = useRef(null);
  const audioCtxRef    = useRef(null);
  const processorRef   = useRef(null);
  const analyserRef    = useRef(null);
  const rafRef         = useRef(null);
  const audioPlayerRef = useRef(null);
  const recordingRef   = useRef(false);
  const watchdogRef    = useRef(null);
  const lastChunkRef   = useRef(0);

  // BUG FIX: websocket and language stored as refs so initAudioPipeline
  // never captures a stale closure — always reads the current value
  const wsRef   = useRef(websocket);
  const langRef = useRef(language);
  useEffect(() => { wsRef.current = websocket; }, [websocket]);
  useEffect(() => { langRef.current = language; }, [language]);

  // auto-play translated audio
  useEffect(() => {
    if (!audioUrl || !audioPlayerRef.current) return;
    audioPlayerRef.current.pause();
    audioPlayerRef.current.src = audioUrl;
    audioPlayerRef.current.load();
    audioPlayerRef.current.play().catch(() => {});
  }, [audioUrl]);

  // cleanup on unmount
  useEffect(() => () => teardownAudio(), []);

  function sendWs(payload) {
    const ws = wsRef.current?.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(payload));
    }
  }

  function teardownAudio() {
    recordingRef.current = false;
    if (watchdogRef.current)   { clearInterval(watchdogRef.current);   watchdogRef.current  = null; }
    if (rafRef.current)        { cancelAnimationFrame(rafRef.current); rafRef.current        = null; }
    if (processorRef.current)  { try { processorRef.current.disconnect(); } catch(e){} processorRef.current = null; }
    if (analyserRef.current)   { try { analyserRef.current.disconnect();  } catch(e){} analyserRef.current  = null; }
    if (mediaStreamRef.current){ mediaStreamRef.current.getTracks().forEach(t => t.stop()); mediaStreamRef.current = null; }
    if (audioCtxRef.current && audioCtxRef.current.state !== "closed") {
      audioCtxRef.current.close().catch(() => {});
      audioCtxRef.current = null;
    }
  }

  function stopRecording() {
    // Fix (Bug C): tell the server to stop/flush the STT engine. This is
    // only sent on an explicit user stop, not from internal pipeline
    // teardowns (e.g. the watchdog restart below calls initAudioPipeline
    // directly, not this function — a transient restart shouldn't flip
    // server-side recording state).
    sendWs({ type: "stop_recording" });
    teardownAudio();
    setRecording(false);
    setVolume(0);
  }

  // BUG FIX: initAudioPipeline reads ws and language from REFS not props
  // This prevents stale closures when pipeline restarts after AudioContext suspend
  async function initAudioPipeline() {
    const ws = wsRef.current?.current;   // wsRef.current = the useRef from App.jsx, .current = the WebSocket
    if (!ws || ws.readyState !== WebSocket.OPEN) return;

    // tear down any existing pipeline first
    if (watchdogRef.current)  { clearInterval(watchdogRef.current);  watchdogRef.current = null; }
    if (processorRef.current) { try { processorRef.current.disconnect(); } catch(e){} processorRef.current = null; }
    if (analyserRef.current)  { try { analyserRef.current.disconnect();  } catch(e){} analyserRef.current  = null; }
    if (mediaStreamRef.current){ mediaStreamRef.current.getTracks().forEach(t => t.stop()); mediaStreamRef.current = null; }
    if (audioCtxRef.current && audioCtxRef.current.state !== "closed") {
      audioCtxRef.current.close().catch(() => {});
      audioCtxRef.current = null;
    }

    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount:     1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl:  true,
        }
      });
    } catch (err) {
      setMicError(
        err.name === "NotAllowedError"
          ? "Microphone blocked. Allow access in browser settings."
          : `Mic error: ${err.message}`
      );
      recordingRef.current = false;
      setRecording(false);
      return;
    }

    // verify track is actually live before proceeding
    const tracks = stream.getAudioTracks();
    if (!tracks.length || tracks[0].readyState !== "live") {
      setMicError("Microphone track not live. Try clicking Speak again.");
      recordingRef.current = false;
      setRecording(false);
      return;
    }

    mediaStreamRef.current = stream;

    const ctx = new AudioContext();
    audioCtxRef.current = ctx;
    if (ctx.state === "suspended") await ctx.resume();

    const source  = ctx.createMediaStreamSource(stream);

    // volume meter
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 256;
    analyserRef.current = analyser;
    source.connect(analyser);

    const volBuf = new Uint8Array(analyser.frequencyBinCount);
    function tickVolume() {
      if (!recordingRef.current) return;
      analyser.getByteFrequencyData(volBuf);
      const avg = volBuf.reduce((a, b) => a + b, 0) / volBuf.length;
      setVolume(Math.min(100, avg * 2));
      rafRef.current = requestAnimationFrame(tickVolume);
    }
    rafRef.current = requestAnimationFrame(tickVolume);

    // PCM capture
    const processor = ctx.createScriptProcessor(CHUNK_SAMPLES, 1, 1);
    processorRef.current = processor;

    processor.onaudioprocess = (evt) => {
      if (!recordingRef.current) return;
      const currentWs = wsRef.current?.current;   // always read fresh
      if (!currentWs || currentWs.readyState !== WebSocket.OPEN) return;

      lastChunkRef.current = Date.now();

      const float32 = evt.inputBuffer.getChannelData(0);
      let samples   = float32;

      if (ctx.sampleRate !== SAMPLE_RATE) {
        const ratio     = ctx.sampleRate / SAMPLE_RATE;
        const outLen    = Math.round(float32.length / ratio);
        const resampled = new Float32Array(outLen);
        for (let i = 0; i < outLen; i++) {
          resampled[i] = float32[Math.min(Math.round(i * ratio), float32.length - 1)];
        }
        samples = resampled;
      }

      const int16 = new Int16Array(samples.length);
      for (let i = 0; i < samples.length; i++) {
        const s  = Math.max(-1, Math.min(1, samples[i]));
        int16[i] = s < 0 ? s * 32768 : s * 32767;
      }
      currentWs.send(int16.buffer);
    };

    source.connect(processor);
    processor.connect(ctx.destination);

    // watchdog — detects AudioContext suspension and restarts
    lastChunkRef.current = Date.now();
    watchdogRef.current = setInterval(async () => {
      if (!recordingRef.current) return;

      // resume suspended context (happens when audio plays)
      if (audioCtxRef.current && audioCtxRef.current.state === "suspended") {
        try { await audioCtxRef.current.resume(); } catch(e) {}
        return;
      }

      // if no chunks for 4s despite recording, full restart
      if (Date.now() - lastChunkRef.current > 4000) {
        console.warn(`[${userId}] No audio chunks — restarting pipeline`);
        const currentWs = wsRef.current?.current;
        if (currentWs && currentWs.readyState === WebSocket.OPEN) {
          currentWs.send(JSON.stringify({ type: "mic_restart" }));
        }
        await initAudioPipeline();
      }
    }, 2000);
  }

  async function startRecording() {
    if (!sessionActive || !connected) return;
    setMicError(null);
    recordingRef.current = true;
    setRecording(true);
    // Fix (Bug C): tell the server to start its STT engine BEFORE we
    // begin the (slower) mic pipeline setup below. This still isn't a
    // perfect guarantee of ordering, but it moves the server's engine
    // start as close as possible to actual audio flow instead of firing
    // it on connect, minutes before the user ever presses Speak.
    sendWs({ type: "start_recording" });
    await initAudioPipeline();
  }

  function toggleRecording() {
    if (recording) stopRecording();
    else startRecording();
  }

  const isActive = sessionActive && connected;

  // Fix (Bug B): label the transcript with the language it was actually
  // transcribed in, falling back to the current dropdown selection only
  // when there's no transcript yet (nothing to mislabel).
  const transcriptText = transcript?.text || "";
  const transcriptLang = transcript?.language || language;

  return (
    <div className={`panel ${recording ? "panel-recording" : ""}`}>

      <div className="panel-header">
        <div className="panel-title">
          <span className="panel-user-id">{userId}</span>
          <span className="panel-label">{label}</span>
        </div>
        <div className={`connection-dot ${connected ? "dot-green" : "dot-gray"}`} />
      </div>

      <div className="panel-section">
        <label className="field-label">Speaking language</label>
        <select
          className="lang-select"
          value={language}
          onChange={e => {
            const newLang = e.target.value;
            setLanguage(newLang);
            sendWs({ type: "language_change", language: newLang });
          }}
          disabled={recording}
        >
          {languages.map(l => (
            <option key={l.code} value={l.code}>{l.label}</option>
          ))}
        </select>
      </div>

      <div className="panel-section center">
        <button
          className={`mic-btn ${recording ? "mic-btn-active" : ""} ${!isActive ? "mic-btn-disabled" : ""}`}
          onClick={toggleRecording}
          disabled={!isActive}
        >
          <span className="mic-icon">{recording ? "⏹" : "🎙"}</span>
          <span className="mic-label">{recording ? "Stop" : "Speak"}</span>
        </button>

        {recording && (
          <div className="volume-bar-wrap">
            <div className="volume-bar" style={{ width: `${volume}%` }} />
          </div>
        )}
        {micError && <p className="error-text">{micError}</p>}
      </div>

      <div className="panel-section">
        <label className="field-label">{label} said ({transcriptLang}):</label>
        <div className="text-box">
          {transcriptText || <span className="placeholder">Transcript appears here...</span>}
        </div>
      </div>

      <div className="panel-section">
        <label className="field-label">Translated for {targetLabel}:</label>
        <div className="text-box text-box-translated">
          {translation || <span className="placeholder">Translation appears here...</span>}
        </div>
      </div>

      <div className="panel-section">
        <label className="field-label">Audio for {targetLabel}</label>
        <audio ref={audioPlayerRef} controls className="audio-player" />
      </div>

    </div>
  );
}
