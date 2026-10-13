import { useState, useMemo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Bug, Search, Trash2, Edit3 } from 'lucide-react'
import { useStore } from '../store/useStore'
import { useI18n } from '../hooks/useI18n'
import { PageHeader, Card, SeverityBadge, StatusBadge, Modal, EmptyState } from '../components/UI'
import { findingsAPI } from '../api'

const SEVERITIES = ['critical', 'high', 'medium', 'low', 'informational', 'unknown']
const STATUSES = ['new', 'under_review', 'confirmed', 'false_positive', 'accepted_risk', 'resolved']

export default function Findings() {
  const { t } = useI18n()
  const findings = useStore((s) => s.findings)
  const fetchFindings = useStore((s) => s.fetchFindings)
  const [search, setSearch] = useState('')
  const [sevFilter, setSevFilter] = useState('')
  const [statusFilter, setStatusFilter] = useState('')
  const [selected, setSelected] = useState(null)
  const [editNotes, setEditNotes] = useState('')

  const filtered = useMemo(() => {
    return findings.filter((f) => {
      if (search && !f.title.toLowerCase().includes(search.toLowerCase()) && !f.description.toLowerCase().includes(search.toLowerCase())) return false
      if (sevFilter && f.severity !== sevFilter) return false
      if (statusFilter && f.validation_status !== statusFilter) return false
      return true
    })
  }, [findings, search, sevFilter, statusFilter])

  const updateStatus = async (id, status) => {
    await findingsAPI.update(id, { validation_status: status, review_status: status })
    fetchFindings()
    setSelected({ ...selected, validation_status: status, review_status: status })
  }

  const saveNotes = async () => {
    await findingsAPI.update(selected.id, { notes: editNotes })
    fetchFindings()
    setSelected({ ...selected, notes: editNotes })
  }

  const del = async (id) => {
    if (!confirm('Delete this finding?')) return
    await findingsAPI.delete(id)
    fetchFindings()
    setSelected(null)
  }

  return (
    <div>
      <PageHeader title={t('findings')} subtitle="Normalized security findings from assessments" />

      {/* Filters */}
      <Card className="mb-4 !p-4">
        <div className="flex flex-wrap gap-3">
          <div className="relative flex-1 min-w-[200px]">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-text-secondary" />
            <input className="input-field pl-10" placeholder={t('search')} value={search} onChange={(e) => setSearch(e.target.value)} />
          </div>
          <select className="input-field w-40" value={sevFilter} onChange={(e) => setSevFilter(e.target.value)}>
            <option value="">All Severities</option>
            {SEVERITIES.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
          <select className="input-field w-40" value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)}>
            <option value="">All Statuses</option>
            {STATUSES.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
        </div>
      </Card>

      {filtered.length === 0 ? (
        <EmptyState message="No findings match your filters" />
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filtered.map((f, i) => (
            <Card key={f.id} delay={i * 0.03}>
              <div className="flex items-start justify-between mb-2">
                <div className="flex items-center gap-2">
                  <Bug className="w-4 h-4 text-accent-primary" />
                  {f.is_demo && <span className="badge bg-warning/15 text-warning border border-warning/30">{t('demo')}</span>}
                </div>
                <SeverityBadge severity={f.severity} />
              </div>
              <h3 className="text-sm font-medium text-text-primary mb-1 cursor-pointer hover:text-accent-primary" onClick={() => { setSelected(f); setEditNotes(f.notes || '') }}>{f.title}</h3>
              <p className="text-xs text-text-secondary line-clamp-2 mb-2">{f.description}</p>
              <div className="flex items-center justify-between text-[10px] text-text-secondary">
                <span>{f.source_tool || '—'}</span>
                <StatusBadge status={f.validation_status} />
              </div>
            </Card>
          ))}
        </div>
      )}

      {/* Detail Modal */}
      <AnimatePresence>
        {selected && (
          <Modal isOpen={!!selected} onClose={() => setSelected(null)} title={selected.title} maxWidth="max-w-2xl">
            <div className="space-y-4">
              <div className="flex items-center gap-3">
                <SeverityBadge severity={selected.severity} />
                <StatusBadge status={selected.validation_status} />
                <span className="text-xs text-text-secondary">Confidence: {selected.confidence}</span>
                {selected.is_demo && <span className="badge bg-warning/15 text-warning border border-warning/30">{t('demo')}</span>}
              </div>

              <div>
                <label className="text-xs text-text-secondary mb-1 block">{t('description')}</label>
                <p className="text-sm text-text-primary bg-bg-secondary rounded-lg p-3">{selected.description}</p>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-xs text-text-secondary mb-1 block">{t('target')}</label>
                  <p className="text-sm text-text-primary">{selected.target}</p>
                </div>
                <div>
                  <label className="text-xs text-text-secondary mb-1 block">Affected Component</label>
                  <p className="text-sm text-text-primary">{selected.affected_component || '—'}</p>
                </div>
                <div>
                  <label className="text-xs text-text-secondary mb-1 block">{t('sourceTool')}</label>
                  <p className="text-sm text-text-primary">{selected.source_tool || '—'}</p>
                </div>
                <div>
                  <label className="text-xs text-text-secondary mb-1 block">Source Agent</label>
                  <p className="text-sm text-text-primary">{selected.source_agent || '—'}</p>
                </div>
              </div>

              <div>
                <label className="text-xs text-text-secondary mb-1 block">{t('evidence')}</label>
                <pre className="text-xs text-text-primary bg-bg-secondary rounded-lg p-3 overflow-x-auto whitespace-pre-wrap font-mono">{selected.evidence || 'No evidence recorded'}</pre>
              </div>

              <div>
                <label className="text-xs text-text-secondary mb-1 block">{t('remediation')}</label>
                <p className="text-sm text-text-primary bg-bg-secondary rounded-lg p-3">{selected.remediation || 'No remediation guidance'}</p>
              </div>

              {/* Status change */}
              <div>
                <label className="text-xs text-text-secondary mb-1 block">Change Review Status</label>
                <div className="flex flex-wrap gap-2">
                  {STATUSES.map((s) => (
                    <button
                      key={s}
                      onClick={() => updateStatus(selected.id, s)}
                      className={`badge border transition-colors ${selected.validation_status === s ? 'bg-accent-primary/15 text-accent-primary border-accent-primary/30' : 'bg-bg-secondary text-text-secondary border-border hover:border-accent-primary/30'}`}
                    >
                      {s.replace('_', ' ')}
                    </button>
                  ))}
                </div>
              </div>

              {/* Notes */}
              <div>
                <label className="text-xs text-text-secondary mb-1 block">Notes</label>
                <textarea className="input-field" rows="2" value={editNotes} onChange={(e) => setEditNotes(e.target.value)} />
                <button onClick={saveNotes} className="btn-secondary text-xs mt-2">Save Notes</button>
              </div>

              <div className="flex justify-between pt-2">
                <button onClick={() => del(selected.id)} className="text-error text-sm flex items-center gap-1 hover:underline">
                  <Trash2 className="w-3.5 h-3.5" /> Delete
                </button>
                <button onClick={() => setSelected(null)} className="btn-secondary text-sm">{t('close')}</button>
              </div>
            </div>
          </Modal>
        )}
      </AnimatePresence>
    </div>
  )
}
