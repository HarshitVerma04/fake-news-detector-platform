import { useState, useEffect, useCallback } from 'react'
import { predictArticle, fetchHistory } from './api'

// ─────────────────────────────────────────────
// Utilities
// ─────────────────────────────────────────────

function formatDate(iso) {
  const d = new Date(iso)
  return d.toLocaleString('en-GB', {
    day: '2-digit', month: 'short', year: 'numeric',
    hour: '2-digit', minute: '2-digit',
  })
}

function truncate(str, max = 120) {
  if (!str) return ''
  return str.length > max ? str.slice(0, max) + '…' : str
}

// ─────────────────────────────────────────────
// ConfidenceMeter
// ─────────────────────────────────────────────

function ConfidenceMeter({ confidence, prediction }) {
  const pct = Math.round(confidence * 100)
  const color = prediction === 'Fake' ? 'var(--fake)' : 'var(--real)'

  return (
    <div style={styles.meterWrap}>
      <div style={styles.meterLabel}>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--text-secondary)', letterSpacing: '0.08em', textTransform: 'uppercase' }}>
          Confidence
        </span>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '20px', color, fontWeight: 500 }}>
          {pct}%
        </span>
      </div>
      <div style={styles.meterTrack}>
        <div
          style={{
            ...styles.meterFill,
            width: `${pct}%`,
            background: color,
            boxShadow: `0 0 12px ${color}66`,
          }}
        />
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────
// ResultCard
// ─────────────────────────────────────────────

function ResultCard({ result }) {
  const isFake = result.prediction === 'Fake'
  const colorVar = isFake ? 'var(--fake)' : 'var(--real)'
  const bgVar = isFake ? 'var(--fake-bg)' : 'var(--real-bg)'
  const borderVar = isFake ? 'var(--fake-border)' : 'var(--real-border)'

  return (
    <div className="fade-in" style={{ ...styles.resultCard, background: bgVar, border: `1px solid ${borderVar}` }}>
      <div style={styles.resultHeader}>
        <div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--text-muted)', letterSpacing: '0.12em', textTransform: 'uppercase', marginBottom: '6px' }}>
            Analysis Result
          </div>
          <div style={{ fontFamily: 'var(--font-display)', fontSize: '42px', fontWeight: 900, color: colorVar, lineHeight: 1, letterSpacing: '-0.02em' }}>
            {result.prediction}
          </div>
        </div>
        <div style={{ ...styles.resultBadge, background: colorVar }}>
          {isFake ? '✕' : '✓'}
        </div>
      </div>

      <ConfidenceMeter confidence={result.confidence} prediction={result.prediction} />

      <div style={styles.resultMeta}>
        <span style={styles.metaItem}>
          <span style={styles.metaLabel}>Record ID</span>
          <span style={styles.metaValue}>#{result.id}</span>
        </span>
        <span style={styles.metaItem}>
          <span style={styles.metaLabel}>Analyzed</span>
          <span style={styles.metaValue}>{formatDate(result.created_at)}</span>
        </span>
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────
// HistoryRow
// ─────────────────────────────────────────────

function HistoryRow({ item, index }) {
  const isFake = item.prediction === 'Fake'
  return (
    <div
      className="fade-in"
      style={{
        ...styles.historyRow,
        animationDelay: `${index * 0.04}s`,
        opacity: 0,
      }}
    >
      <div style={{ display: 'flex', alignItems: 'flex-start', gap: '14px', flex: 1, minWidth: 0 }}>
        <div style={{
          ...styles.historyBadge,
          background: isFake ? 'var(--fake-bg)' : 'var(--real-bg)',
          color: isFake ? 'var(--fake)' : 'var(--real)',
          border: `1px solid ${isFake ? 'var(--fake-border)' : 'var(--real-border)'}`,
        }}>
          {item.prediction}
        </div>
        <div style={{ minWidth: 0 }}>
          <div style={styles.historyText}>{truncate(item.news_text)}</div>
          <div style={styles.historyDate}>{formatDate(item.created_at)}</div>
        </div>
      </div>
      <div style={{
        fontFamily: 'var(--font-mono)',
        fontSize: '13px',
        color: isFake ? 'var(--fake)' : 'var(--real)',
        whiteSpace: 'nowrap',
        paddingLeft: '12px',
      }}>
        {Math.round(item.confidence * 100)}%
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────
// HistoryPanel
// ─────────────────────────────────────────────

function HistoryPanel({ history, loading, onRefresh }) {
  if (loading) {
    return (
      <div style={styles.historyEmpty}>
        <div style={styles.spinner} />
        <span style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', fontSize: '12px' }}>
          Loading history...
        </span>
      </div>
    )
  }

  if (!history || history.items.length === 0) {
    return (
      <div style={styles.historyEmpty}>
        <div style={{ fontSize: '28px', marginBottom: '8px', opacity: 0.3 }}>◈</div>
        <span style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', fontSize: '12px' }}>
          No analyses yet
        </span>
      </div>
    )
  }

  return (
    <div>
      <div style={styles.historyHeaderRow}>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--text-muted)', letterSpacing: '0.1em' }}>
          {history.total} TOTAL RECORDS
        </span>
        <button onClick={onRefresh} style={styles.refreshBtn}>
          ↻ Refresh
        </button>
      </div>
      <div style={styles.historyList}>
        {history.items.map((item, i) => (
          <HistoryRow key={item.id} item={item} index={i} />
        ))}
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────
// Main App
// ─────────────────────────────────────────────

export default function App() {
  const [text, setText] = useState('')
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)

  const [history, setHistory] = useState(null)
  const [historyLoading, setHistoryLoading] = useState(false)
  const [activeTab, setActiveTab] = useState('analyze')

  const loadHistory = useCallback(async () => {
    setHistoryLoading(true)
    try {
      const data = await fetchHistory(20, 0)
      setHistory(data)
    } catch (e) {
      console.error('History fetch failed:', e)
    } finally {
      setHistoryLoading(false)
    }
  }, [])

  useEffect(() => {
    loadHistory()
  }, [loadHistory])

  const handleSubmit = async () => {
    if (!text.trim() || text.trim().length < 50) {
      setError('Article must be at least 50 characters.')
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const data = await predictArticle(text.trim())
      setResult(data)
      loadHistory()
    } catch (e) {
      setError(e.message || 'Something went wrong. Is the backend running?')
    } finally {
      setLoading(false)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      handleSubmit()
    }
  }

  const charCount = text.length
  const isReady = charCount >= 50

  return (
    <div style={styles.root}>
      {/* Header */}
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
              <div style={styles.statLabel}>Test Accuracy</div>
            </div>
            <div style={styles.statDivider} />
            <div style={styles.statBlock}>
              <div style={styles.statValue}>TF-IDF</div>
              <div style={styles.statLabel}>+ LogReg</div>
            </div>
            <div style={styles.statDivider} />
            <div style={styles.statBlock}>
              <div style={styles.statValue}>72K</div>
              <div style={styles.statLabel}>Training Articles</div>
            </div>
          </div>
        </div>
      </header>

      {/* Tab Bar */}
      <div style={styles.tabBar}>
        <div style={styles.tabBarInner}>
          {['analyze', 'history'].map(tab => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              style={{
                ...styles.tab,
                ...(activeTab === tab ? styles.tabActive : {}),
              }}
            >
              {tab === 'analyze' ? '◈ Analyze Article' : `◇ History${history ? ` (${history.total})` : ''}`}
            </button>
          ))}
        </div>
      </div>

      {/* Main Content */}
      <main style={styles.main}>
        {activeTab === 'analyze' && (
          <div className="fade-in" style={styles.analyzeLayout}>

            {/* Input Panel */}
            <div style={styles.inputPanel}>
              <label style={styles.inputLabel}>
                <span>Article Text</span>
                <span style={{
                  fontFamily: 'var(--font-mono)',
                  fontSize: '11px',
                  color: isReady ? 'var(--accent)' : 'var(--text-muted)',
                }}>
                  {charCount} chars{!isReady && ` · ${50 - charCount} more needed`}
                </span>
              </label>

              <textarea
                value={text}
                onChange={e => { setText(e.target.value); setError(null) }}
                onKeyDown={handleKeyDown}
                placeholder="Paste a news article here to analyze whether it is real or fake. The model evaluates writing patterns, vocabulary, and linguistic structure…"
                style={styles.textarea}
                rows={12}
              />

              {error && (
                <div style={styles.errorMsg}>
                  ⚠ {error}
                </div>
              )}

              <button
                onClick={handleSubmit}
                disabled={loading || !isReady}
                style={{
                  ...styles.submitBtn,
                  ...(loading || !isReady ? styles.submitBtnDisabled : {}),
                }}
              >
                {loading ? (
                  <span style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <span style={styles.spinner} /> Analyzing…
                  </span>
                ) : (
                  'Analyze Article  ⌘↵'
                )}
              </button>

              <div style={styles.hint}>
                Tip: Paste a full article for best results. Short headlines may give low confidence scores.
              </div>
            </div>

            {/* Result Panel */}
            <div style={styles.resultPanel}>
              {result ? (
                <ResultCard result={result} />
              ) : (
                <div style={styles.resultPlaceholder}>
                  <div style={styles.placeholderIcon}>◈</div>
                  <div style={styles.placeholderTitle}>Awaiting Analysis</div>
                  <div style={styles.placeholderText}>
                    Paste article text on the left and click Analyze. The model will classify it as Fake or Real with a confidence score.
                  </div>
                  <div style={styles.placeholderDivider} />
                  <div style={styles.placeholderNote}>
                    Model: TF-IDF + Logistic Regression<br />
                    Dataset: WELFake (72,134 articles)<br />
                    Test F1: 0.9727
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab === 'history' && (
          <div className="fade-in" style={styles.historyPanel}>
            <HistoryPanel
              history={history}
              loading={historyLoading}
              onRefresh={loadHistory}
            />
          </div>
        )}
      </main>

      {/* Footer */}
      <footer style={styles.footer}>
        <span>Verity · Fake News Detection Platform</span>
        <span style={{ color: 'var(--border-light)' }}>·</span>
        <span>TF-IDF + Logistic Regression · WELFake Dataset</span>
        <span style={{ color: 'var(--border-light)' }}>·</span>
        <span>FastAPI · PostgreSQL · React</span>
      </footer>
    </div>
  )
}

// ─────────────────────────────────────────────
// Styles
// ─────────────────────────────────────────────

const styles = {
  root: {
    minHeight: '100vh',
    display: 'flex',
    flexDirection: 'column',
    background: 'var(--bg-primary)',
  },

  // Header
  header: {
    borderBottom: '1px solid var(--border)',
    background: 'var(--bg-secondary)',
  },
  headerInner: {
    maxWidth: '1200px',
    margin: '0 auto',
    padding: '32px 40px 28px',
    display: 'flex',
    alignItems: 'center',
    gap: '40px',
  },
  headerEyebrow: {
    fontFamily: 'var(--font-mono)',
    fontSize: '10px',
    letterSpacing: '0.18em',
    color: 'var(--accent)',
    marginBottom: '6px',
  },
  headerTitle: {
    fontFamily: 'var(--font-display)',
    fontSize: '52px',
    fontWeight: 900,
    letterSpacing: '-0.03em',
    color: 'var(--text-primary)',
    lineHeight: 1,
  },
  headerSub: {
    fontFamily: 'var(--font-mono)',
    fontSize: '11px',
    color: 'var(--text-muted)',
    letterSpacing: '0.1em',
    marginTop: '6px',
    textTransform: 'uppercase',
  },
  headerRule: {
    width: '1px',
    height: '60px',
    background: 'var(--border)',
    flexShrink: 0,
  },
  headerStats: {
    display: 'flex',
    alignItems: 'center',
    gap: '24px',
  },
  statBlock: {
    textAlign: 'center',
  },
  statValue: {
    fontFamily: 'var(--font-mono)',
    fontSize: '18px',
    fontWeight: 500,
    color: 'var(--accent)',
    letterSpacing: '-0.02em',
  },
  statLabel: {
    fontFamily: 'var(--font-mono)',
    fontSize: '10px',
    color: 'var(--text-muted)',
    letterSpacing: '0.08em',
    textTransform: 'uppercase',
    marginTop: '2px',
  },
  statDivider: {
    width: '1px',
    height: '32px',
    background: 'var(--border)',
  },

  // Tab bar
  tabBar: {
    borderBottom: '1px solid var(--border)',
    background: 'var(--bg-secondary)',
  },
  tabBarInner: {
    maxWidth: '1200px',
    margin: '0 auto',
    padding: '0 40px',
    display: 'flex',
    gap: '0',
  },
  tab: {
    background: 'none',
    border: 'none',
    borderBottom: '2px solid transparent',
    padding: '14px 20px',
    fontFamily: 'var(--font-mono)',
    fontSize: '12px',
    letterSpacing: '0.06em',
    color: 'var(--text-muted)',
    cursor: 'pointer',
    transition: 'color var(--transition), border-color var(--transition)',
    textTransform: 'uppercase',
  },
  tabActive: {
    color: 'var(--accent)',
    borderBottomColor: 'var(--accent)',
  },

  // Main
  main: {
    flex: 1,
    maxWidth: '1200px',
    margin: '0 auto',
    padding: '40px',
    width: '100%',
  },

  // Analyze layout
  analyzeLayout: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: '32px',
    alignItems: 'start',
  },

  // Input panel
  inputPanel: {
    display: 'flex',
    flexDirection: 'column',
    gap: '14px',
  },
  inputLabel: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    fontFamily: 'var(--font-mono)',
    fontSize: '11px',
    letterSpacing: '0.1em',
    textTransform: 'uppercase',
    color: 'var(--text-secondary)',
  },
  textarea: {
    width: '100%',
    background: 'var(--bg-input)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius)',
    color: 'var(--text-primary)',
    fontFamily: 'var(--font-body)',
    fontSize: '15px',
    lineHeight: '1.7',
    padding: '18px',
    resize: 'vertical',
    outline: 'none',
    transition: 'border-color var(--transition)',
    caretColor: 'var(--accent)',
  },
  errorMsg: {
    fontFamily: 'var(--font-mono)',
    fontSize: '12px',
    color: 'var(--fake)',
    padding: '10px 14px',
    background: 'var(--fake-bg)',
    border: '1px solid var(--fake-border)',
    borderRadius: 'var(--radius)',
  },
  submitBtn: {
    background: 'var(--accent)',
    color: '#0f0e0c',
    border: 'none',
    borderRadius: 'var(--radius)',
    padding: '14px 24px',
    fontFamily: 'var(--font-mono)',
    fontSize: '13px',
    fontWeight: 500,
    letterSpacing: '0.06em',
    cursor: 'pointer',
    transition: 'opacity var(--transition), transform var(--transition)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '8px',
  },
  submitBtnDisabled: {
    opacity: 0.35,
    cursor: 'not-allowed',
    transform: 'none',
  },
  hint: {
    fontFamily: 'var(--font-mono)',
    fontSize: '11px',
    color: 'var(--text-muted)',
    lineHeight: 1.5,
  },

  // Result panel
  resultPanel: {
    position: 'sticky',
    top: '24px',
  },
  resultCard: {
    borderRadius: 'var(--radius)',
    padding: '28px',
    display: 'flex',
    flexDirection: 'column',
    gap: '24px',
  },
  resultHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
  },
  resultBadge: {
    width: '44px',
    height: '44px',
    borderRadius: '50%',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: '20px',
    fontWeight: 700,
    color: '#fff',
    flexShrink: 0,
  },
  resultMeta: {
    display: 'flex',
    gap: '24px',
    paddingTop: '16px',
    borderTop: '1px solid var(--border)',
  },
  metaItem: {
    display: 'flex',
    flexDirection: 'column',
    gap: '2px',
  },
  metaLabel: {
    fontFamily: 'var(--font-mono)',
    fontSize: '10px',
    color: 'var(--text-muted)',
    textTransform: 'uppercase',
    letterSpacing: '0.1em',
  },
  metaValue: {
    fontFamily: 'var(--font-mono)',
    fontSize: '13px',
    color: 'var(--text-secondary)',
  },

  // Confidence meter
  meterWrap: {
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
  },
  meterLabel: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  meterTrack: {
    height: '4px',
    background: 'var(--border)',
    borderRadius: '2px',
    overflow: 'hidden',
  },
  meterFill: {
    height: '100%',
    borderRadius: '2px',
    transition: 'width 0.6s ease',
  },

  // Placeholder
  resultPlaceholder: {
    background: 'var(--bg-card)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius)',
    padding: '40px 32px',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    textAlign: 'center',
    gap: '12px',
  },
  placeholderIcon: {
    fontSize: '36px',
    color: 'var(--border-light)',
    lineHeight: 1,
  },
  placeholderTitle: {
    fontFamily: 'var(--font-display)',
    fontSize: '22px',
    color: 'var(--text-secondary)',
    fontWeight: 700,
  },
  placeholderText: {
    fontFamily: 'var(--font-body)',
    fontSize: '14px',
    color: 'var(--text-muted)',
    lineHeight: 1.7,
    maxWidth: '280px',
  },
  placeholderDivider: {
    width: '40px',
    height: '1px',
    background: 'var(--border)',
    margin: '4px 0',
  },
  placeholderNote: {
    fontFamily: 'var(--font-mono)',
    fontSize: '11px',
    color: 'var(--text-muted)',
    lineHeight: 2,
    letterSpacing: '0.04em',
  },

  // History
  historyPanel: {
    maxWidth: '860px',
  },
  historyHeaderRow: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '16px',
    paddingBottom: '12px',
    borderBottom: '1px solid var(--border)',
  },
  refreshBtn: {
    background: 'none',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius)',
    color: 'var(--text-muted)',
    fontFamily: 'var(--font-mono)',
    fontSize: '11px',
    padding: '5px 12px',
    cursor: 'pointer',
    letterSpacing: '0.06em',
    transition: 'color var(--transition), border-color var(--transition)',
  },
  historyList: {
    display: 'flex',
    flexDirection: 'column',
    gap: '1px',
  },
  historyRow: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '16px 18px',
    background: 'var(--bg-card)',
    borderRadius: 'var(--radius)',
    gap: '12px',
    transition: 'background var(--transition)',
  },
  historyBadge: {
    fontFamily: 'var(--font-mono)',
    fontSize: '10px',
    letterSpacing: '0.1em',
    textTransform: 'uppercase',
    padding: '3px 8px',
    borderRadius: '2px',
    whiteSpace: 'nowrap',
    flexShrink: 0,
  },
  historyText: {
    fontFamily: 'var(--font-body)',
    fontSize: '14px',
    color: 'var(--text-secondary)',
    lineHeight: 1.5,
    marginBottom: '4px',
  },
  historyDate: {
    fontFamily: 'var(--font-mono)',
    fontSize: '11px',
    color: 'var(--text-muted)',
    letterSpacing: '0.04em',
  },
  historyEmpty: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: '10px',
    padding: '60px 0',
    color: 'var(--text-muted)',
  },

  // Spinner
  spinner: {
    width: '14px',
    height: '14px',
    border: '2px solid currentColor',
    borderTopColor: 'transparent',
    borderRadius: '50%',
    display: 'inline-block',
    animation: 'spin 0.7s linear infinite',
    flexShrink: 0,
  },

  // Footer
  footer: {
    borderTop: '1px solid var(--border)',
    padding: '16px 40px',
    display: 'flex',
    gap: '12px',
    alignItems: 'center',
    fontFamily: 'var(--font-mono)',
    fontSize: '11px',
    color: 'var(--text-muted)',
    letterSpacing: '0.06em',
  },
}
