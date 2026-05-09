import { useEffect, useMemo, useRef, useState } from 'react';
import { AlertTriangle, Brain, CircleDashed, Send, ShieldCheck, Activity, Sparkles } from 'lucide-react';
import { AnimatePresence, motion } from 'framer-motion';
import axios from 'axios';

const API_BASE = import.meta.env.VITE_DADBOT_API_BASE_URL ?? 'http://127.0.0.1:8010/v1';
const DADBOT_SECRET_KEY = import.meta.env.VITE_DADBOT_SECRET_KEY ?? '';
const SESSION_ID = 'zero-g-demo';
const TENANT_ID = 'family-a';
const FIXTURE_MODE =
  import.meta.env.VITE_DADBOT_USE_FIXTURES === '1' ||
  new URLSearchParams(window.location.search).get('fixtures') === '1';

const seedMessages = [
  {
    id: 'm1',
    role: 'system',
    text: 'Zero-G HUD online. Waiting for kernel telemetry.',
    time: '00:00:00',
  },
];

const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

function meterLabel(level) {
  if (level >= 80) return 'stable';
  if (level >= 55) return 'warm';
  if (level >= 30) return 'watch';
  return 'critical';
}

function ConnectionChip({ state }) {
  const tone = state === 'open' ? 'ok' : state === 'connecting' ? 'warn' : 'bad';
  return <span className={`chip chip-${tone}`}>{state}</span>;
}

function VitalGauge({ label, value, accent, suffix = '%' }) {
  const safeValue = clamp(Number(value) || 0, 0, 100);
  return (
    <div className="vital-gauge">
      <div className="vital-gauge__header">
        <span>{label}</span>
        <strong>{safeValue.toFixed(0)}{suffix}</strong>
      </div>
      <div className="meter-track">
        <motion.div
          className="meter-fill"
          style={{ background: accent }}
          initial={false}
          animate={{ width: `${safeValue}%` }}
          transition={{ type: 'spring', stiffness: 180, damping: 24 }}
        />
      </div>
    </div>
  );
}

function FragmentCard({ fragment, index }) {
  return (
    <motion.div
      className="fragment-card"
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.04 }}
    >
      <div className="fragment-card__top">
        <span>{fragment.memory_id || fragment.fragment_id || `fragment-${index + 1}`}</span>
        <strong>{Math.round((fragment.similarity_score || 0) * 100)}%</strong>
      </div>
      <p>{fragment.summary || 'No summary provided.'}</p>
      <div className="fragment-card__meta">
        <span>{fragment.category || 'uncategorized'}</span>
        <span>{fragment.source || 'subconscious_reflex'}</span>
      </div>
      {fragment.checksum ? <code>{fragment.checksum}</code> : null}
    </motion.div>
  );
}

function MessageRow({ message }) {
  return (
    <div className={`message-row message-row-${message.role}`}>
      <div className="message-row__meta">
        <span>{message.role}</span>
        <span>{message.time}</span>
      </div>
      <div className="message-row__body">{message.text}</div>
    </div>
  );
}

export default function App() {
  const [messages, setMessages] = useState(seedMessages);
  const [draft, setDraft] = useState('');
  const [sessionId] = useState(SESSION_ID);
  const [connectionState, setConnectionState] = useState('idle');
  const [pulse, setPulse] = useState(null);
  const [lastTurn, setLastTurn] = useState(null);
  const [pulseError, setPulseError] = useState('');
  const [turnError, setTurnError] = useState('');
  const [fixtureTurn, setFixtureTurn] = useState(null);
  const [fixturePulse, setFixturePulse] = useState(null);
  const [fixtureReady, setFixtureReady] = useState(false);
  const logEndRef = useRef(null);
  const socketRef = useRef(null);
  const reconnectTimerRef = useRef(null);

  const pulseData = pulse?.trust_meter ?? {
    trust_level: 50,
    openness_level: 50,
    trust_credit: 0.5,
    alignment_score: 0.5,
    label: 'steady',
  };

  const driftData = pulse?.drift_alarm ?? {
    active: false,
    current_risk_level: 'low',
    predicted_drift_probability: 0,
    likely_trigger_category: 'unknown',
    confidence_score: 0,
    recommended_intervention: '',
    intervention_justification: '',
  };

  const integrityData = pulse?.integrity_status ?? { merkle_check_passed: true, reason: '' };

  const inferredStatus = useMemo(() => {
    if (driftData.active) return 'drift watch';
    if ((pulse?.current_thoughts || []).length) return 'thinking';
    return pulse?.status || 'idle';
  }, [driftData.active, pulse]);

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    if (!FIXTURE_MODE) return;
    let cancelled = false;

    async function loadFixtures() {
      try {
        const [turnResp, pulseResp] = await Promise.all([
          fetch('/fixtures/fixture_turn.json'),
          fetch('/fixtures/fixture_pulse.json'),
        ]);
        const [turnPayload, pulsePayload] = await Promise.all([turnResp.json(), pulseResp.json()]);
        if (cancelled) return;
        setFixtureTurn(turnPayload);
        setFixturePulse(pulsePayload);
        setPulse(pulsePayload);
        setConnectionState('open');
        setPulseError('');
        setFixtureReady(true);
      } catch (error) {
        if (cancelled) return;
        setFixtureReady(false);
        setConnectionState('error');
        setPulseError(`Fixture load failed: ${String(error)}`);
      }
    }

    loadFixtures();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (FIXTURE_MODE) {
      return undefined;
    }
    let cancelled = false;

    const connect = () => {
      if (cancelled) return;
      setConnectionState('connecting');
      const protocol = API_BASE.startsWith('https') ? 'wss' : 'ws';
      const wsBase = API_BASE.replace(/^https?/, protocol).replace(/\/v1\/?$/, '');
      const query = new URLSearchParams({ tenant_id: TENANT_ID });
      if (DADBOT_SECRET_KEY) {
        query.set('dadbot_key', DADBOT_SECRET_KEY);
      }
      const socketUrl = `${wsBase}/v1/sessions/${sessionId}/pulse/stream?${query.toString()}`;
      const socket = new WebSocket(socketUrl);
      socketRef.current = socket;

      socket.onopen = () => {
        setConnectionState('open');
        setPulseError('');
      };

      socket.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data);
          if (payload.event_type === 'pulse.heartbeat') return;
          setPulse(payload);
        } catch (error) {
          setPulseError(String(error));
        }
      };

      socket.onerror = () => {
        setConnectionState('error');
      };

      socket.onclose = () => {
        setConnectionState('closed');
        if (!cancelled) {
          reconnectTimerRef.current = window.setTimeout(connect, 1500);
        }
      };
    };

    connect();

    return () => {
      cancelled = true;
      if (reconnectTimerRef.current) {
        window.clearTimeout(reconnectTimerRef.current);
      }
      socketRef.current?.close();
    };
  }, [sessionId]);

  async function submitTurn(event) {
    event.preventDefault();
    const text = draft.trim();
    if (!text) return;

    setTurnError('');
    setDraft('');
    const userMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      text,
      time: new Date().toLocaleTimeString([], { hour12: false }),
    };
    setMessages((current) => [...current, userMessage]);

    if (FIXTURE_MODE) {
      const turn = fixtureTurn;
      const replyText =
        turn?.response_text ||
        'Fixture mode active. Kernel reply stream is mocked from local JSON payloads.';
      setLastTurn(turn || null);
      setPulse((current) =>
        current
          ? {
              ...current,
              subconscious_metadata: turn?.subconscious_metadata ?? current.subconscious_metadata,
              integrity_status: turn?.integrity_status ?? current.integrity_status,
              trust_meter: turn?.trust_meter ?? current.trust_meter,
              drift_alarm: turn?.drift_alarm ?? current.drift_alarm,
            }
          : fixturePulse,
      );
      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: 'assistant',
          text: replyText,
          time: new Date().toLocaleTimeString([], { hour12: false }),
        },
      ]);
      return;
    }

    try {
      const response = await axios.post(
        `${API_BASE}/sessions/${sessionId}/turn`,
        {
          user_input: text,
          tenant_id: TENANT_ID,
          timeout_seconds: 45,
        },
        {
          headers: {
            'Content-Type': 'application/json',
            ...(DADBOT_SECRET_KEY ? { 'X-DADBOT-KEY': DADBOT_SECRET_KEY } : {}),
          },
        },
      );

      const turn = response.data?.turn;
      const replyText = turn?.response_text || response.data?.response?.reply || 'No response text returned.';
      setLastTurn(turn || null);
      setPulse((current) =>
        current
          ? {
              ...current,
              subconscious_metadata: turn?.subconscious_metadata ?? current.subconscious_metadata,
              integrity_status: turn?.integrity_status ?? current.integrity_status,
              trust_meter: turn?.trust_meter ?? current.trust_meter,
              drift_alarm: turn?.drift_alarm ?? current.drift_alarm,
            }
          : current,
      );
      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: 'assistant',
          text: replyText,
          time: new Date().toLocaleTimeString([], { hour12: false }),
        },
      ]);
    } catch (error) {
      const message = axios.isAxiosError(error)
        ? error.response?.data?.detail || error.message
        : String(error);
      setTurnError(message);
      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: 'system',
          text: `Turn failed: ${message}`,
          time: new Date().toLocaleTimeString([], { hour12: false }),
        },
      ]);
    }
  }

  return (
    <div className="hud-shell">
      <div className="hud-noise" />
      <header className="hud-topbar">
        <div>
          <div className="eyebrow">DadBot Zero-G HUD</div>
          <h1>Kernel Broadcast Console</h1>
        </div>
        <div className="topbar-status">
          <ConnectionChip state={connectionState} />
          {FIXTURE_MODE ? (
            <span className="chip chip-warn">
              <Sparkles size={14} /> fixtures
            </span>
          ) : null}
          <span className="chip chip-neutral">{inferredStatus}</span>
          <span className="chip chip-neutral">{API_BASE}</span>
        </div>
      </header>

      <main className="hud-grid">
        <section className="terminal-panel">
          <div className="panel-header">
            <div>
              <div className="eyebrow">Main Log</div>
              <h2>Monospaced conversation feed</h2>
            </div>
            <div className="panel-header__badges">
              <span className="chip chip-neutral">session {sessionId}</span>
              {lastTurn?.integrity_status?.merkle_check_passed ? (
                <span className="chip chip-ok"><ShieldCheck size={14} /> merkle ok</span>
              ) : (
                <span className="chip chip-bad"><AlertTriangle size={14} /> integrity risk</span>
              )}
            </div>
          </div>

          <div className="terminal-feed">
            <AnimatePresence initial={false}>
              {messages.map((message) => (
                <motion.div
                  key={message.id}
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -8 }}
                >
                  <MessageRow message={message} />
                </motion.div>
              ))}
            </AnimatePresence>
            <div ref={logEndRef} />
          </div>

          <form className="composer" onSubmit={submitTurn}>
            <label className="composer__label" htmlFor="prompt">
              <CircleDashed size={14} />
              Transmit prompt
            </label>
            <textarea
              id="prompt"
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              placeholder="Ask the kernel something direct..."
              rows={4}
            />
            <div className="composer__footer">
              <div className="composer__status">
                {turnError ? (
                  <span className="error-text">{turnError}</span>
                ) : FIXTURE_MODE ? (
                  <span>{fixtureReady ? 'Fixture mode active. Local contract payloads are driving the HUD.' : 'Fixture mode booting...'}</span>
                ) : (
                  <span>Terminal-first feed. UI stays dumb.</span>
                )}
              </div>
              <button type="submit" className="send-button">
                <Send size={16} />
                Send
              </button>
            </div>
          </form>
        </section>

        <aside className="sidebar-panel">
          <div className="panel-header">
            <div>
              <div className="eyebrow">Subconscious</div>
              <h2>Retrieved fragments</h2>
            </div>
            <span className="chip chip-neutral">{pulse?.subconscious_metadata?.length ?? 0} fragments</span>
          </div>

          <div className="fragments-list">
            {(pulse?.subconscious_metadata || []).length > 0 ? (
              pulse.subconscious_metadata.map((fragment, index) => (
                <FragmentCard key={`${fragment.fragment_id || fragment.memory_id || index}`} fragment={fragment} index={index} />
              ))
            ) : (
              <div className="empty-state">
                <Brain size={20} />
                <p>No fragments broadcast yet. The kernel will fill this once retrieval activates.</p>
              </div>
            )}
          </div>
        </aside>

        <section className="vitals-panel">
          <div className="panel-header">
            <div>
              <div className="eyebrow">Vitals HUD</div>
              <h2>Pulse stream telemetry</h2>
            </div>
            <span className={`chip ${integrityData.merkle_check_passed ? 'chip-ok' : 'chip-bad'}`}>
              <Activity size={14} /> {integrityData.merkle_check_passed ? 'integrity stable' : 'integrity fault'}
            </span>
          </div>

          <div className="vitals-stack">
            <VitalGauge label="Trust" value={pulseData.trust_level} accent="linear-gradient(90deg, #f59e0b, #f97316)" />
            <VitalGauge label="Openness" value={pulseData.openness_level} accent="linear-gradient(90deg, #60a5fa, #22d3ee)" />
            <VitalGauge label="Drift" value={clamp((driftData.predicted_drift_probability || 0) * 100, 0, 100)} accent="linear-gradient(90deg, #f97316, #ef4444)" />
          </div>

          <div className="status-card">
            <div className="status-card__row">
              <span>Meter state</span>
              <strong>{pulseData.label || meterLabel(pulseData.trust_level)}</strong>
            </div>
            <div className="status-card__row">
              <span>Inference intent</span>
              <strong>{pulse?.inference_intent || 'Idle'}</strong>
            </div>
            <div className="status-card__row">
              <span>Drift alarm</span>
              <strong>{driftData.active ? 'active' : 'quiet'}</strong>
            </div>
            <div className="status-card__row">
              <span>Merkle check</span>
              <strong>{integrityData.merkle_check_passed ? 'passed' : 'failed'}</strong>
            </div>
          </div>

          <div className="drift-card">
            <div className="drift-card__title">
              <AlertTriangle size={16} />
              Drift signal
            </div>
            <p>{driftData.recommended_intervention || 'No active drift recommendation.'}</p>
            <small>{driftData.intervention_justification || 'The kernel has not emitted a drift justification yet.'}</small>
          </div>

          {pulseError ? <div className="error-text">Pulse socket: {pulseError}</div> : null}
        </section>
      </main>
    </div>
  );
}
