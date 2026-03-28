import { useState, useEffect, useCallback, createContext, useContext } from 'react'
import {
  predictArticle, fetchFromUrl, fetchHistory,
  fetchStats, fetchModelInfo,
  registerUser, loginUser, getMe, requestPasswordReset, confirmPasswordReset,
} from './api'

// ─────────────────────────────────────────────
// Auth Context
// ─────────────────────────────────────────────

const AuthContext = createContext(null)

function AuthProvider({ children }) {
  const [user, setUser]   = useState(null)
  const [token, setToken] = useState(() => localStorage.getItem('access_token'))
  const [ready, setReady] = useState(false)

  useEffect(() => {
    if (token) {
      getMe(token)
        .then(setUser)
        .catch(() => { localStorage.removeItem('access_token'); setToken(null) })
        .finally(() => setReady(true))
    } else {
      setReady(true)
    }
  }, [token])

  const login = useCallback(async (creds) => {
    const data = await loginUser(creds)
    localStorage.setItem('access_token', data.access_token)
    localStorage.setItem('refresh_token', data.refresh_token)
    setToken(data.access_token)
    setUser(data.user)
  }, [])

  const logout = useCallback(() => {
    localStorage.removeItem('access_token')
    localStorage.removeItem('refresh_token')
    setToken(null)
    setUser(null)
  }, [])

  return (
    <AuthContext.Provider value={{ user, token, login, logout, ready }}>
      {ready ? children : <div style={styles.splash}>Loading…</div>}
    </AuthContext.Provider>
  )
}

function useAuth() { return useContext(AuthContext) }

// ─────────────────────────────────────────────
// Utilities
// ─────────────────────────────────────────────

function formatDate(iso) {
  return new Date(iso).toLocaleString('en-GB', {
    day: '2-digit', month: 'short', year: 'numeric',
    hour: '2-digit', minute: '2-digit',
  })
}
function truncate(str, max = 110) {
  return !str ? '' : str.length > max ? str.slice(0, max) + '…' : str
}

// ─────────────────────────────────────────────
// Shared UI components
// ─────────────────────────────────────────────

function ModelBadge({ model }) {
  const isBert = model === 'distilbert'
  return (
    <span style={{
      fontFamily: 'var(--font-mono)', fontSize: '10px', letterSpacing: '0.1em',
      textTransform: 'uppercase', padding: '3px 8px', borderRadius: '2px',
      background: isBert ? 'rgba(100,149,237,0.12)' : 'var(--accent-glow)',
      color: isBert ? '#6495ed' : 'var(--accent)',
      border: `1px solid ${isBert ? 'rgba(100,149,237,0.3)' : 'rgba(212,168,71,0.3)'}`,
    }}>
      {isBert ? 'DistilBERT' : 'TF-IDF + LR'}
    </span>
  )
}

function Spinner({ size = 14 }) {
  return <span style={{ width: size, height: size, border: '2px solid currentColor', borderTopColor: 'transparent', borderRadius: '50%', display: 'inline-block', animation: 'spin 0.7s linear infinite', flexShrink: 0 }} />
}

function ErrorMsg({ msg }) {
  return msg ? <div style={styles.errorMsg}>⚠ {msg}</div> : null
}

// ─────────────────────────────────────────────
// Auth Modal
// ─────────────────────────────────────────────

function AuthModal({ onClose }) {
  const { login } = useAuth()
  const [mode, setMode]         = useState('login')   // login | register | reset | resetConfirm
  const [form, setForm]         = useState({})
  const [loading, setLoading]   = useState(false)
  const [error, setError]       = useState(null)
  const [success, setSuccess]   = useState(null)

  const set = (k, v) => setForm(f => ({ ...f, [k]: v }))

  const handleSubmit = async () => {
    setLoading(true); setError(null); setSuccess(null)
    try {
      if (mode === 'login') {
        await login({ username: form.username, password: form.password })
        onClose()
      } else if (mode === 'register') {
        await registerUser({ username: form.username, email: form.email, password: form.password })
        setSuccess('Account created! You can now log in.')
        setMode('login')
      } else if (mode === 'reset') {
        const res = await requestPasswordReset(form.email)
        setSuccess(`Reset token: ${res.debug_token}`)
        setMode('resetConfirm')
      } else if (mode === 'resetConfirm') {
        await confirmPasswordReset({ token: form.resetToken, new_password: form.newPassword })
        setSuccess('Password updated. Please log in.')
        setMode('login')
      }
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const titles = { login: 'Sign In', register: 'Create Account', reset: 'Reset Password', resetConfirm: 'Set New Password' }

  return (
    <div style={styles.modalOverlay} onClick={e => e.target === e.currentTarget && onClose()}>
      <div style={styles.modal}>
        <div style={styles.modalHeader}>
          <div style={styles.modalTitle}>{titles[mode]}</div>
          <button onClick={onClose} style={styles.closeBtn}>✕</button>
        </div>

        {success && <div style={styles.successMsg}>{success}</div>}
        <ErrorMsg msg={error} />

        <div style={styles.formFields}>
          {(mode === 'login' || mode === 'register') && (
            <input placeholder="Username or email" value={form.username || ''} onChange={e => set('username', e.target.value)} style={styles.input} />
          )}
          {mode === 'register' && (
            <input placeholder="Email address" type="email" value={form.email || ''} onChange={e => set('email', e.target.value)} style={styles.input} />
          )}
          {(mode === 'login' || mode === 'register') && (
            <input placeholder="Password (min 8 chars)" type="password" value={form.password || ''} onChange={e => set('password', e.target.value)} style={styles.input} onKeyDown={e => e.key === 'Enter' && handleSubmit()} />
          )}
          {mode === 'reset' && (
            <input placeholder="Your email address" type="email" value={form.email || ''} onChange={e => set('email', e.target.value)} style={styles.input} />
          )}
          {mode === 'resetConfirm' && (<>
            <input placeholder="Reset token (from email)" value={form.resetToken || ''} onChange={e => set('resetToken', e.target.value)} style={styles.input} />
            <input placeholder="New password" type="password" value={form.newPassword || ''} onChange={e => set('newPassword', e.target.value)} style={styles.input} />
          </>)}
        </div>

        <button onClick={handleSubmit} disabled={loading} style={{ ...styles.submitBtn, width: '100%', justifyContent: 'center', opacity: loading ? 0.5 : 1 }}>
          {loading ? <Spinner /> : titles[mode]}
        </button>

        <div style={styles.authLinks}>
          {mode === 'login' && (<>
            <button onClick={() => setMode('register')} style={styles.linkBtn}>Create account</button>
            <button onClick={() => setMode('reset')} style={styles.linkBtn}>Forgot password?</button>
          </>)}
          {mode !== 'login' && <button onClick={() => setMode('login')} style={styles.linkBtn}>Back to sign in</button>}
        </div>
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────
// ConfidenceMeter
// ─────────────────────────────────────────────

function ConfidenceMeter({ confidence, prediction }) {
  const pct   = Math.round(confidence * 100)
  const color = prediction === 'Fake' ? 'var(--fake)' : 'var(--real)'
  return (
    <div style={styles.meterWrap}>
      <div style={styles.meterLabel}>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--text-secondary)', letterSpacing: '0.08em', textTransform: 'uppercase' }}>Confidence</span>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '20px', color, fontWeight: 500 }}>{pct}%</span>
      </div>
      <div style={styles.meterTrack}>
        <div style={{ ...styles.meterFill, width: `${pct}%`, background: color, boxShadow: `0 0 12px ${color}66` }} />
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────
// ResultCard
// ─────────────────────────────────────────────

function ResultCard({ result }) {
  const isFake = result.prediction === 'Fake'
  const color  = isFake ? 'var(--fake)' : 'var(--real)'
  const bg     = isFake ? 'var(--fake-bg)' : 'var(--real-bg)'
  const border = isFake ? 'var(--fake-border)' : 'var(--real-border)'

  return (
    <div className="fade-in" style={{ ...styles.resultCard, background: bg, border: `1px solid ${border}` }}>
      <div style={styles.resultHeader}>
        <div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--text-muted)', letterSpacing: '0.12em', textTransform: 'uppercase', marginBottom: '6px' }}>Analysis Result</div>
          <div style={{ fontFamily: 'var(--font-display)', fontSize: '42px', fontWeight: 900, color, lineHeight: 1, letterSpacing: '-0.02em' }}>{result.prediction}</div>
        </div>
        <div style={{ ...styles.resultBadge, background: color }}>{isFake ? '✕' : '✓'}</div>
      </div>
      <ConfidenceMeter confidence={result.confidence} prediction={result.prediction} />
      {result.source_url && (
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--text-muted)' }}>
          Source: <a href={result.source_url} target="_blank" rel="noreferrer" style={{ color: 'var(--accent)', textDecoration: 'none' }}>{truncate(result.source_url, 60)}</a>
        </div>
      )}
      <div style={styles.resultMeta}>
        <span style={styles.metaItem}><span style={styles.metaLabel}>ID</span><span style={styles.metaValue}>#{result.id}</span></span>
        <span style={styles.metaItem}><span style={styles.metaLabel}>Model</span><ModelBadge model={result.model_used} /></span>
        <span style={styles.metaItem}><span style={styles.metaLabel}>Analyzed</span><span style={styles.metaValue}>{formatDate(result.created_at)}</span></span>
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────
// Analyze Tab
// ─────────────────────────────────────────────

function AnalyzeTab({ modelInfo }) {
  const { token } = useAuth()
  const [mode, setMode]       = useState('text')   // 'text' | 'url'
  const [text, setText]       = useState('')
  const [url, setUrl]         = useState('')
  const [result, setResult]   = useState(null)
  const [error, setError]     = useState(null)
  const [loading, setLoading] = useState(false)

  const handleSubmit = async () => {
    setLoading(true); setError(null); setResult(null)
    try {
      if (mode === 'text') {
        if (text.trim().length < 50) { setError('Article must be at least 50 characters.'); return }
        setResult(await predictArticle(text.trim(), token))
      } else {
        if (!url.trim().startsWith('http')) { setError('Please enter a valid URL starting with http.'); return }
        setResult(await fetchFromUrl(url.trim(), token))
      }
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const charCount = text.length
  const isReady   = mode === 'url' ? url.startsWith('http') : charCount >= 50

  return (
    <div style={styles.analyzeLayout}>
      <div style={styles.inputPanel}>
        {/* Mode toggle */}
        <div style={styles.modeToggle}>
          {['text', 'url'].map(m => (
            <button key={m} onClick={() => { setMode(m); setError(null); setResult(null) }}
              style={{ ...styles.modeBtn, ...(mode === m ? styles.modeBtnActive : {}) }}>
              {m === 'text' ? '⌨ Paste Text' : '🔗 From URL'}
            </button>
          ))}
        </div>

        <label style={styles.inputLabel}>
          <span>{mode === 'text' ? 'Article Text' : 'Article URL'}</span>
          {mode === 'text' && (
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: isReady ? 'var(--accent)' : 'var(--text-muted)' }}>
              {charCount} chars{!isReady && ` · ${50 - charCount} more needed`}
            </span>
          )}
        </label>

        {mode === 'text' ? (
          <textarea
            value={text}
            onChange={e => { setText(e.target.value); setError(null) }}
            onKeyDown={e => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) handleSubmit() }}
            placeholder="Paste a news article here…"
            style={styles.textarea}
            rows={12}
          />
        ) : (
          <input
            value={url}
            onChange={e => { setUrl(e.target.value); setError(null) }}
            onKeyDown={e => e.key === 'Enter' && handleSubmit()}
            placeholder="https://www.bbc.com/news/..."
            style={{ ...styles.textarea, height: '56px', resize: 'none', padding: '14px 18px', fontFamily: 'var(--font-mono)', fontSize: '13px' }}
          />
        )}

        <ErrorMsg msg={error} />

        <button onClick={handleSubmit} disabled={loading || !isReady}
          style={{ ...styles.submitBtn, ...(loading || !isReady ? styles.submitBtnDisabled : {}) }}>
          {loading
            ? <span style={{ display: 'flex', alignItems: 'center', gap: '10px' }}><Spinner /> {mode === 'url' ? 'Fetching & analyzing…' : 'Analyzing…'}</span>
            : mode === 'url' ? 'Fetch & Analyze' : 'Analyze Article  ⌘↵'
          }
        </button>

        {modelInfo && (
          <div style={styles.hint}>
            Using: <ModelBadge model={modelInfo.active_model} />
          </div>
        )}
      </div>

      <div style={styles.resultPanel}>
        {result ? <ResultCard result={result} /> : (
          <div style={styles.resultPlaceholder}>
            <div style={styles.placeholderIcon}>◈</div>
            <div style={styles.placeholderTitle}>Awaiting Analysis</div>
            <div style={styles.placeholderText}>
              {mode === 'text' ? 'Paste article text and click Analyze.' : 'Enter a news article URL and click Fetch & Analyze.'}
            </div>
            <div style={styles.placeholderDivider} />
            <div style={styles.placeholderNote}>
              Model: {modelInfo?.active_model === 'distilbert' ? 'DistilBERT' : 'TF-IDF + LogReg'}<br />
              Dataset: WELFake (72,134 articles)<br />
              Test F1: {modelInfo?.distilbert_metrics?.f1 ?? '0.9727'}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────
// History Tab
// ─────────────────────────────────────────────

function HistoryTab() {
  const { token } = useAuth()
  const [history, setHistory]   = useState(null)
  const [loading, setLoading]   = useState(false)

  const load = useCallback(async () => {
    setLoading(true)
    try { setHistory(await fetchHistory(20, 0, token)) }
    catch (e) { console.error(e) }
    finally { setLoading(false) }
  }, [token])

  useEffect(() => { load() }, [load])

  if (loading) return <div style={styles.emptyState}><Spinner /></div>
  if (!history || history.items.length === 0) return (
    <div style={styles.emptyState}>
      <div style={{ fontSize: '28px', opacity: 0.3 }}>◈</div>
      <span style={{ fontFamily: 'var(--font-mono)', fontSize: '12px', color: 'var(--text-muted)' }}>No analyses yet</span>
    </div>
  )

  return (
    <div style={{ maxWidth: '860px' }}>
      <div style={styles.historyHeaderRow}>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--text-muted)', letterSpacing: '0.1em' }}>
          {history.total} RECORDS
        </span>
        <button onClick={load} style={styles.refreshBtn}>↻ Refresh</button>
      </div>
      <div style={styles.historyList}>
        {history.items.map((item, i) => {
          const isFake = item.prediction === 'Fake'
          return (
            <div key={item.id} className="fade-in" style={{ ...styles.historyRow, animationDelay: `${i * 0.03}s`, opacity: 0 }}>
              <div style={{ display: 'flex', alignItems: 'flex-start', gap: '14px', flex: 1, minWidth: 0 }}>
                <div style={{ ...styles.historyBadge, background: isFake ? 'var(--fake-bg)' : 'var(--real-bg)', color: isFake ? 'var(--fake)' : 'var(--real)', border: `1px solid ${isFake ? 'var(--fake-border)' : 'var(--real-border)'}` }}>
                  {item.prediction}
                </div>
                <div style={{ minWidth: 0 }}>
                  <div style={styles.historyText}>{item.source_url ? <a href={item.source_url} target="_blank" rel="noreferrer" style={{ color: 'var(--accent)', textDecoration: 'none' }}>{truncate(item.source_url, 80)}</a> : truncate(item.news_text)}</div>
                  <div style={{ display: 'flex', gap: '10px', alignItems: 'center', marginTop: '4px' }}>
                    <span style={styles.historyDate}>{formatDate(item.created_at)}</span>
                    <ModelBadge model={item.model_used} />
                  </div>
                </div>
              </div>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '13px', color: isFake ? 'var(--fake)' : 'var(--real)', whiteSpace: 'nowrap' }}>
                {Math.round(item.confidence * 100)}%
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────
// Stats Tab — pure SVG charts
// ─────────────────────────────────────────────

function DonutChart({ fake, real }) {
  const total = fake + real
  if (total === 0) return null
  const fakePct = fake / total
  const r = 56, cx = 70, cy = 70
  const circumference = 2 * Math.PI * r
  const fakeArc = circumference * fakePct

  return (
    <svg width="140" height="140" viewBox="0 0 140 140">
      <circle cx={cx} cy={cy} r={r} fill="none" stroke="var(--fake)" strokeWidth="18" strokeDasharray={`${fakeArc} ${circumference - fakeArc}`} strokeDashoffset={circumference * 0.25} />
      <circle cx={cx} cy={cy} r={r} fill="none" stroke="var(--real)" strokeWidth="18" strokeDasharray={`${circumference - fakeArc} ${fakeArc}`} strokeDashoffset={circumference * 0.25 - fakeArc} />
      <text x={cx} y={cy - 6} textAnchor="middle" fill="var(--text-primary)" fontFamily="var(--font-mono)" fontSize="16" fontWeight="600">{Math.round(fakePct * 100)}%</text>
      <text x={cx} y={cy + 14} textAnchor="middle" fill="var(--text-muted)" fontFamily="var(--font-mono)" fontSize="10">FAKE</text>
    </svg>
  )
}

function BarChart({ data, valueKey, labelKey, color = 'var(--accent)' }) {
  if (!data || data.length === 0) return null
  const max = Math.max(...data.map(d => d[valueKey]), 1)
  const W = 460, H = 120, pad = 32, barW = Math.max(8, (W - pad * 2) / data.length - 4)

  return (
    <svg width={W} height={H + 28} viewBox={`0 0 ${W} ${H + 28}`} style={{ overflow: 'visible' }}>
      {data.map((d, i) => {
        const barH = Math.max(2, (d[valueKey] / max) * H)
        const x = pad + i * ((W - pad * 2) / data.length)
        return (
          <g key={i}>
            <rect x={x} y={H - barH} width={barW} height={barH} fill={color} rx="2" opacity="0.85" />
            <text x={x + barW / 2} y={H + 16} textAnchor="middle" fill="var(--text-muted)" fontFamily="var(--font-mono)" fontSize="9">
              {d[labelKey]?.slice(-5)}
            </text>
          </g>
        )
      })}
    </svg>
  )
}

function StatsTab() {
  const [stats, setStats]     = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError]     = useState(null)

  useEffect(() => {
    fetchStats()
      .then(setStats)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={styles.emptyState}><Spinner /></div>
  if (error)   return <div style={styles.emptyState}><ErrorMsg msg={error} /></div>
  if (!stats || stats.total_analyses === 0) return (
    <div style={styles.emptyState}>
      <div style={{ fontSize: '28px', opacity: 0.3 }}>◈</div>
      <span style={{ fontFamily: 'var(--font-mono)', fontSize: '12px', color: 'var(--text-muted)' }}>
        No data yet — run some analyses first
      </span>
    </div>
  )

  const modelEntries = Object.entries(stats.model_breakdown)

  return (
    <div className="fade-in" style={styles.statsGrid}>

      {/* Summary cards */}
      <div style={styles.statsRow}>
        {[
          { label: 'Total Analyses', value: stats.total_analyses, color: 'var(--accent)' },
          { label: 'Fake Articles',  value: `${stats.fake_count} (${stats.fake_pct}%)`, color: 'var(--fake)' },
          { label: 'Real Articles',  value: `${stats.real_count} (${stats.real_pct}%)`, color: 'var(--real)' },
          { label: 'Avg Confidence', value: `${stats.avg_confidence}%`, color: 'var(--text-primary)' },
        ].map(card => (
          <div key={card.label} style={styles.statCard}>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', color: 'var(--text-muted)', letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: '8px' }}>{card.label}</div>
            <div style={{ fontFamily: 'var(--font-display)', fontSize: '28px', fontWeight: 700, color: card.color }}>{card.value}</div>
          </div>
        ))}
      </div>

      {/* Donut + model breakdown */}
      <div style={styles.statsRowHalf}>
        <div style={styles.chartCard}>
          <div style={styles.chartTitle}>Fake vs Real Distribution</div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '32px' }}>
            <DonutChart fake={stats.fake_count} real={stats.real_count} />
            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              {[['Fake', 'var(--fake)', stats.fake_count], ['Real', 'var(--real)', stats.real_count]].map(([label, color, count]) => (
                <div key={label} style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <div style={{ width: '10px', height: '10px', borderRadius: '2px', background: color, flexShrink: 0 }} />
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: '12px', color: 'var(--text-secondary)' }}>{label}: {count}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div style={styles.chartCard}>
          <div style={styles.chartTitle}>Model Usage</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            {modelEntries.map(([model, count]) => {
              const pct = Math.round(count / stats.total_analyses * 100)
              return (
                <div key={model}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
                    <ModelBadge model={model} />
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: '12px', color: 'var(--text-secondary)' }}>{count} ({pct}%)</span>
                  </div>
                  <div style={{ height: '4px', background: 'var(--border)', borderRadius: '2px' }}>
                    <div style={{ height: '100%', width: `${pct}%`, background: model === 'distilbert' ? '#6495ed' : 'var(--accent)', borderRadius: '2px', transition: 'width 0.6s ease' }} />
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      </div>

      {/* Daily chart */}
      {stats.daily_counts.length > 0 && (
        <div style={styles.chartCard}>
          <div style={styles.chartTitle}>Analyses — Last 30 Days</div>
          <BarChart data={stats.daily_counts} valueKey="fake" labelKey="date" color="var(--fake)" />
        </div>
      )}

      {/* Confidence buckets */}
      <div style={styles.chartCard}>
        <div style={styles.chartTitle}>Confidence Distribution</div>
        <BarChart data={stats.confidence_buckets} valueKey="count" labelKey="range" color="var(--accent)" />
        <div style={{ display: 'flex', gap: '16px', marginTop: '12px', flexWrap: 'wrap' }}>
          {stats.confidence_buckets.map(b => (
            <div key={b.range} style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--text-muted)' }}>
              {b.range}: <span style={{ color: 'var(--text-secondary)' }}>{b.count}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────
// Models Tab
// ─────────────────────────────────────────────

function ModelsTab({ modelInfo }) {
  if (!modelInfo) return <div style={styles.emptyState}><Spinner /></div>
  const t = modelInfo.tfidf_metrics
  const b = modelInfo.distilbert_metrics

  return (
    <div className="fade-in" style={{ maxWidth: '720px' }}>
      <div style={styles.modelsHeader}>
        <div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', color: 'var(--text-muted)', letterSpacing: '0.12em', textTransform: 'uppercase', marginBottom: '4px' }}>Active Model</div>
          <ModelBadge model={modelInfo.active_model} />
        </div>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--text-muted)' }}>Test set · WELFake dataset</div>
      </div>

      <div style={styles.comparisonTable}>
        <div style={styles.comparisonHeader}>
          <span style={styles.metricLabel} />
          <span style={{ ...styles.metricVal, color: 'var(--accent)', fontWeight: 500 }}>TF-IDF + LR</span>
          <span style={{ ...styles.metricVal, color: '#6495ed', fontWeight: 500 }}>DistilBERT</span>
        </div>
        {['accuracy', 'precision', 'recall', 'f1'].map(metric => {
          const tv = t?.[metric], bv = b?.[metric]
          return (
            <div key={metric} style={styles.metricRow}>
              <span style={styles.metricLabel}>{metric.charAt(0).toUpperCase() + metric.slice(1)}</span>
              <span style={styles.metricVal}>{tv != null ? `${(tv * 100).toFixed(2)}%` : '—'}</span>
              <span style={{ ...styles.metricVal, color: bv != null && tv != null && bv > tv ? 'var(--real)' : bv != null ? 'var(--text-secondary)' : 'var(--text-muted)' }}>
                {bv != null ? `${(bv * 100).toFixed(2)}%` : 'Not trained'}{bv != null && tv != null && bv > tv && <span style={{ marginLeft: '6px', fontSize: '10px' }}>▲</span>}
              </span>
            </div>
          )
        })}
      </div>

      {!b && (
        <div style={styles.bertNote}>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--accent)', marginBottom: '8px', letterSpacing: '0.08em' }}>DISTILBERT NOT TRAINED YET</div>
          <pre style={styles.codeBlock}>{`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets accelerate
python scripts/train_distilbert.py`}</pre>
          <div style={{ fontFamily: 'var(--font-body)', fontSize: '13px', color: 'var(--text-muted)', lineHeight: 1.6 }}>
            Then set <code style={{ color: 'var(--accent)', fontFamily: 'var(--font-mono)', fontSize: '12px' }}>ACTIVE_MODEL=distilbert</code> in your <code style={{ color: 'var(--accent)', fontFamily: 'var(--font-mono)', fontSize: '12px' }}>.env</code> and restart the server.
          </div>
        </div>
      )}
    </div>
  )
}

// ─────────────────────────────────────────────
// App Shell
// ─────────────────────────────────────────────

function AppShell() {
  const { user, logout }            = useAuth()
  const [activeTab, setActiveTab]   = useState('analyze')
  const [showAuth, setShowAuth]     = useState(false)
  const [modelInfo, setModelInfo]   = useState(null)

  useEffect(() => {
    fetchModelInfo().then(setModelInfo).catch(console.error)
  }, [])

  const tabs = [
    { id: 'analyze', label: '◈ Analyze' },
    { id: 'history', label: '◇ History' },
    { id: 'stats',   label: '▦ Stats' },
    { id: 'models',  label: '◆ Models' },
  ]

  return (
    <div style={styles.root}>
      {showAuth && <AuthModal onClose={() => setShowAuth(false)} />}

      <header style={styles.header}>
        <div style={styles.headerInner}>
          <div>
            <div style={styles.headerEyebrow}>NLP · MACHINE LEARNING</div>
            <h1 style={styles.headerTitle}>VERITY</h1>
            <div style={styles.headerSub}>Fake News Detection Platform</div>
          </div>
          <div style={styles.headerRule} />
          <div style={styles.headerStats}>
            <div style={styles.statBlock}>
              <div style={styles.statValue}>97.2%</div>
              <div style={styles.statLabel}>TF-IDF Accuracy</div>
            </div>
            <div style={styles.statDivider} />
            <div style={styles.statBlock}>
              <div style={styles.statValue}>{modelInfo ? <ModelBadge model={modelInfo.active_model} /> : '—'}</div>
              <div style={styles.statLabel}>Active Model</div>
            </div>
            <div style={styles.statDivider} />
            <div style={styles.statBlock}>
              <div style={styles.statValue}>72K</div>
              <div style={styles.statLabel}>Training Articles</div>
            </div>
          </div>

          <div style={{ marginLeft: 'auto' }}>
            {user ? (
              <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: '12px', color: 'var(--text-secondary)' }}>
                  {user.username}
                </span>
                <button onClick={logout} style={styles.authBtn}>Sign Out</button>
              </div>
            ) : (
              <button onClick={() => setShowAuth(true)} style={styles.authBtn}>Sign In</button>
            )}
          </div>
        </div>
      </header>

      <div style={styles.tabBar}>
        <div style={styles.tabBarInner}>
          {tabs.map(tab => (
            <button key={tab.id} onClick={() => setActiveTab(tab.id)}
              style={{ ...styles.tab, ...(activeTab === tab.id ? styles.tabActive : {}) }}>
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      <main style={styles.main}>
        {activeTab === 'analyze' && <div className="fade-in"><AnalyzeTab modelInfo={modelInfo} /></div>}
        {activeTab === 'history' && <div className="fade-in"><HistoryTab /></div>}
        {activeTab === 'stats'   && <div className="fade-in"><StatsTab /></div>}
        {activeTab === 'models'  && <div className="fade-in"><ModelsTab modelInfo={modelInfo} /></div>}
      </main>

      <footer style={styles.footer}>
        <span>Verity · Fake News Detection Platform</span>
        <span style={{ color: 'var(--border-light)' }}>·</span>
        <span>FastAPI · PostgreSQL · React</span>
        {user && <><span style={{ color: 'var(--border-light)' }}>·</span><span>Signed in as {user.username}</span></>}
      </footer>
    </div>
  )
}

export default function App() {
  return <AuthProvider><AppShell /></AuthProvider>
}

// ─────────────────────────────────────────────
// Styles
// ─────────────────────────────────────────────

const styles = {
  root: { minHeight: '100vh', display: 'flex', flexDirection: 'column', background: 'var(--bg-primary)' },
  splash: { minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' },
  header: { borderBottom: '1px solid var(--border)', background: 'var(--bg-secondary)' },
  headerInner: { maxWidth: '1200px', margin: '0 auto', padding: '28px 40px', display: 'flex', alignItems: 'center', gap: '40px' },
  headerEyebrow: { fontFamily: 'var(--font-mono)', fontSize: '10px', letterSpacing: '0.18em', color: 'var(--accent)', marginBottom: '6px' },
  headerTitle: { fontFamily: 'var(--font-display)', fontSize: '48px', fontWeight: 900, letterSpacing: '-0.03em', color: 'var(--text-primary)', lineHeight: 1 },
  headerSub: { fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--text-muted)', letterSpacing: '0.1em', marginTop: '6px', textTransform: 'uppercase' },
  headerRule: { width: '1px', height: '60px', background: 'var(--border)', flexShrink: 0 },
  headerStats: { display: 'flex', alignItems: 'center', gap: '24px' },
  statBlock: { textAlign: 'center' },
  statValue: { fontFamily: 'var(--font-mono)', fontSize: '18px', fontWeight: 500, color: 'var(--accent)' },
  statLabel: { fontFamily: 'var(--font-mono)', fontSize: '10px', color: 'var(--text-muted)', letterSpacing: '0.08em', textTransform: 'uppercase', marginTop: '2px' },
  statDivider: { width: '1px', height: '32px', background: 'var(--border)' },
  authBtn: { background: 'none', border: '1px solid var(--border)', borderRadius: 'var(--radius)', color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)', fontSize: '11px', padding: '7px 16px', cursor: 'pointer', letterSpacing: '0.06em', transition: 'border-color 0.18s, color 0.18s' },
  tabBar: { borderBottom: '1px solid var(--border)', background: 'var(--bg-secondary)' },
  tabBarInner: { maxWidth: '1200px', margin: '0 auto', padding: '0 40px', display: 'flex' },
  tab: { background: 'none', border: 'none', borderBottom: '2px solid transparent', padding: '14px 20px', fontFamily: 'var(--font-mono)', fontSize: '12px', letterSpacing: '0.06em', color: 'var(--text-muted)', cursor: 'pointer', transition: 'color 0.18s, border-color 0.18s', textTransform: 'uppercase' },
  tabActive: { color: 'var(--accent)', borderBottomColor: 'var(--accent)' },
  main: { flex: 1, maxWidth: '1200px', margin: '0 auto', padding: '40px', width: '100%' },
  analyzeLayout: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '32px', alignItems: 'start' },
  modeToggle: { display: 'flex', gap: '0', border: '1px solid var(--border)', borderRadius: 'var(--radius)', overflow: 'hidden', width: 'fit-content' },
  modeBtn: { background: 'none', border: 'none', padding: '8px 18px', fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--text-muted)', cursor: 'pointer', letterSpacing: '0.06em', transition: 'background 0.18s, color 0.18s' },
  modeBtnActive: { background: 'var(--accent-glow)', color: 'var(--accent)' },
  inputPanel: { display: 'flex', flexDirection: 'column', gap: '14px' },
  inputLabel: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontFamily: 'var(--font-mono)', fontSize: '11px', letterSpacing: '0.1em', textTransform: 'uppercase', color: 'var(--text-secondary)' },
  textarea: { width: '100%', background: 'var(--bg-input)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', color: 'var(--text-primary)', fontFamily: 'var(--font-body)', fontSize: '15px', lineHeight: '1.7', padding: '18px', resize: 'vertical', outline: 'none', caretColor: 'var(--accent)' },
  errorMsg: { fontFamily: 'var(--font-mono)', fontSize: '12px', color: 'var(--fake)', padding: '10px 14px', background: 'var(--fake-bg)', border: '1px solid var(--fake-border)', borderRadius: 'var(--radius)' },
  successMsg: { fontFamily: 'var(--font-mono)', fontSize: '12px', color: 'var(--real)', padding: '10px 14px', background: 'var(--real-bg)', border: '1px solid var(--real-border)', borderRadius: 'var(--radius)' },
  submitBtn: { background: 'var(--accent)', color: '#0f0e0c', border: 'none', borderRadius: 'var(--radius)', padding: '14px 24px', fontFamily: 'var(--font-mono)', fontSize: '13px', fontWeight: 500, letterSpacing: '0.06em', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' },
  submitBtnDisabled: { opacity: 0.35, cursor: 'not-allowed' },
  hint: { fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '8px' },
  resultPanel: { position: 'sticky', top: '24px' },
  resultCard: { borderRadius: 'var(--radius)', padding: '28px', display: 'flex', flexDirection: 'column', gap: '20px' },
  resultHeader: { display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' },
  resultBadge: { width: '44px', height: '44px', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '20px', fontWeight: 700, color: '#fff', flexShrink: 0 },
  resultMeta: { display: 'flex', gap: '20px', paddingTop: '16px', borderTop: '1px solid var(--border)', flexWrap: 'wrap' },
  metaItem: { display: 'flex', flexDirection: 'column', gap: '4px' },
  metaLabel: { fontFamily: 'var(--font-mono)', fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.1em' },
  metaValue: { fontFamily: 'var(--font-mono)', fontSize: '13px', color: 'var(--text-secondary)' },
  meterWrap: { display: 'flex', flexDirection: 'column', gap: '8px' },
  meterLabel: { display: 'flex', justifyContent: 'space-between', alignItems: 'center' },
  meterTrack: { height: '4px', background: 'var(--border)', borderRadius: '2px', overflow: 'hidden' },
  meterFill: { height: '100%', borderRadius: '2px', transition: 'width 0.6s ease' },
  resultPlaceholder: { background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', padding: '40px 32px', display: 'flex', flexDirection: 'column', alignItems: 'center', textAlign: 'center', gap: '12px' },
  placeholderIcon: { fontSize: '36px', color: 'var(--border-light)', lineHeight: 1 },
  placeholderTitle: { fontFamily: 'var(--font-display)', fontSize: '22px', color: 'var(--text-secondary)', fontWeight: 700 },
  placeholderText: { fontFamily: 'var(--font-body)', fontSize: '14px', color: 'var(--text-muted)', lineHeight: 1.7, maxWidth: '280px' },
  placeholderDivider: { width: '40px', height: '1px', background: 'var(--border)', margin: '4px 0' },
  placeholderNote: { fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--text-muted)', lineHeight: 2 },
  emptyState: { display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '10px', padding: '60px 0' },
  historyHeaderRow: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px', paddingBottom: '12px', borderBottom: '1px solid var(--border)' },
  refreshBtn: { background: 'none', border: '1px solid var(--border)', borderRadius: 'var(--radius)', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', fontSize: '11px', padding: '5px 12px', cursor: 'pointer', letterSpacing: '0.06em' },
  historyList: { display: 'flex', flexDirection: 'column', gap: '1px' },
  historyRow: { display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '16px 18px', background: 'var(--bg-card)', borderRadius: 'var(--radius)', gap: '12px' },
  historyBadge: { fontFamily: 'var(--font-mono)', fontSize: '10px', letterSpacing: '0.1em', textTransform: 'uppercase', padding: '3px 8px', borderRadius: '2px', whiteSpace: 'nowrap', flexShrink: 0 },
  historyText: { fontFamily: 'var(--font-body)', fontSize: '14px', color: 'var(--text-secondary)', lineHeight: 1.5 },
  historyDate: { fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--text-muted)', letterSpacing: '0.04em' },
  statsGrid: { display: 'flex', flexDirection: 'column', gap: '20px', maxWidth: '1000px' },
  statsRow: { display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px' },
  statsRowHalf: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' },
  statCard: { background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', padding: '20px 22px' },
  chartCard: { background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', padding: '22px 24px' },
  chartTitle: { fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--text-muted)', letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: '16px' },
  modelsHeader: { display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '24px', paddingBottom: '16px', borderBottom: '1px solid var(--border)' },
  comparisonTable: { background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', overflow: 'hidden', marginBottom: '24px' },
  comparisonHeader: { display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', padding: '14px 20px', borderBottom: '1px solid var(--border)', background: 'var(--bg-secondary)' },
  metricRow: { display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', padding: '14px 20px', borderBottom: '1px solid var(--border)' },
  metricLabel: { fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--text-muted)', letterSpacing: '0.08em', textTransform: 'uppercase' },
  metricVal: { fontFamily: 'var(--font-mono)', fontSize: '14px', color: 'var(--text-secondary)', textAlign: 'center' },
  bertNote: { background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', padding: '24px', display: 'flex', flexDirection: 'column', gap: '12px' },
  codeBlock: { background: 'var(--bg-input)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', padding: '14px 18px', fontFamily: 'var(--font-mono)', fontSize: '12px', color: 'var(--accent)', lineHeight: 2, overflowX: 'auto' },
  modalOverlay: { position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.75)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000, backdropFilter: 'blur(4px)' },
  modal: { background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', padding: '32px', width: '100%', maxWidth: '420px', display: 'flex', flexDirection: 'column', gap: '16px' },
  modalHeader: { display: 'flex', justifyContent: 'space-between', alignItems: 'center' },
  modalTitle: { fontFamily: 'var(--font-display)', fontSize: '24px', fontWeight: 700, color: 'var(--text-primary)' },
  closeBtn: { background: 'none', border: 'none', color: 'var(--text-muted)', fontSize: '18px', cursor: 'pointer', padding: '4px 8px' },
  formFields: { display: 'flex', flexDirection: 'column', gap: '10px' },
  input: { width: '100%', background: 'var(--bg-input)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', color: 'var(--text-primary)', fontFamily: 'var(--font-body)', fontSize: '14px', padding: '11px 14px', outline: 'none', caretColor: 'var(--accent)' },
  authLinks: { display: 'flex', gap: '16px', justifyContent: 'center' },
  linkBtn: { background: 'none', border: 'none', color: 'var(--accent)', fontFamily: 'var(--font-mono)', fontSize: '11px', cursor: 'pointer', letterSpacing: '0.06em', textDecoration: 'underline' },
  footer: { borderTop: '1px solid var(--border)', padding: '16px 40px', display: 'flex', gap: '12px', alignItems: 'center', fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--text-muted)', letterSpacing: '0.06em' },
}
