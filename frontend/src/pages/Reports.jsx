import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { FileText, Plus, Trash2, Download, Eye } from 'lucide-react'
import { useStore } from '../store/useStore'
import { useI18n } from '../hooks/useI18n'
import { PageHeader, Card, StatusBadge, Modal, EmptyState } from '../components/UI'
import { reportsAPI } from '../api'

export default function Reports() {
  const { t } = useI18n()
  const assessments = useStore((s) => s.assessments)
  const [reports, setReports] = useState([])
  const [selected, setSelected] = useState(null)
  const [generating, setGenerating] = useState(null)
  const [showGen, setShowGen] = useState(false)

  const loadReports = async (assessmentId) => {
    const { data } = await reportsAPI.list(assessmentId)
    setReports(data)
  }

  const generate = async (assessmentId) => {
    setGenerating(assessmentId)
    try {
      const { data } = await reportsAPI.generate(assessmentId)
      setReports((r) => [data, ...r])
      setShowGen(false)
    } catch (e) {
      alert('Error generating report: ' + e.message)
    }
    setGenerating(null)
  }

  const del = async (id) => {
    if (!confirm('Delete this report?')) return
    await reportsAPI.delete(id)
    setReports((r) => r.filter((rep) => rep.id !== id))
  }

  const exportReport = (report) => {
    const content = JSON.stringify(report.content, null, 2)
    const blob = new Blob([content], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${report.title.replace(/\s+/g, '_')}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div>
      <PageHeader
        title={t('reports')}
        subtitle="Generate assessment reports from persisted data"
        action={
          <button onClick={() => { loadReports(); setShowGen(true) }} className="btn-primary flex items-center gap-2 text-sm">
            <Plus className="w-4 h-4" /> Generate Report
          </button>
        }
      />

      {reports.length === 0 ? (
        <Card>
          <EmptyState message="No reports generated yet. Click 'Generate Report' to create one from an assessment." />
        </Card>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {reports.map((r, i) => (
            <Card key={r.id} delay={i * 0.05}>
              <div className="flex items-start justify-between mb-3">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-lg bg-accent-primary/15 flex items-center justify-center">
                    <FileText className="w-5 h-5 text-accent-primary" />
                  </div>
                  <div>
                    <h3 className="text-sm font-medium text-text-primary">{r.title}</h3>
                    <StatusBadge status={r.status} />
                  </div>
                </div>
                {r.is_demo && <span className="badge bg-warning/15 text-warning border border-warning/30">{t('demo')}</span>}
              </div>
              <p className="text-xs text-text-secondary mb-3">
                {r.content?.findings_summary?.total || 0} findings • {r.content?.report_id || '—'}
              </p>
              <div className="flex gap-2">
                <button onClick={() => setSelected(r)} className="btn-secondary text-xs flex-1 flex items-center justify-center gap-1">
                  <Eye className="w-3 h-3" /> View
                </button>
                <button onClick={() => exportReport(r)} className="btn-secondary text-xs flex items-center justify-center gap-1">
                  <Download className="w-3 h-3" /> Export
                </button>
                <button onClick={() => del(r.id)} className="text-text-secondary hover:text-error p-1.5">
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              </div>
            </Card>
          ))}
        </div>
      )}

      {/* Generate modal */}
      <AnimatePresence>
        {showGen && (
          <Modal isOpen={showGen} onClose={() => setShowGen(false)} title="Generate Report">
            <div className="space-y-3">
              <p className="text-sm text-text-secondary">Select an assessment to generate a report:</p>
              {assessments.length === 0 ? (
                <EmptyState message="No assessments available" />
              ) : assessments.map((a) => (
                <div key={a.id} className="flex items-center justify-between p-3 rounded-lg bg-bg-secondary border border-border">
                  <div>
                    <div className="text-sm text-text-primary">{a.name}</div>
                    <div className="text-xs text-text-secondary">{a.target_identifier} • {a.status}</div>
                  </div>
                  <button
                    onClick={() => generate(a.id)}
                    disabled={generating === a.id}
                    className="btn-primary text-xs"
                  >
                    {generating === a.id ? 'Generating...' : 'Generate'}
                  </button>
                </div>
              ))}
            </div>
          </Modal>
        )}
      </AnimatePresence>

      {/* Report viewer */}
      <AnimatePresence>
        {selected && (
          <Modal isOpen={!!selected} onClose={() => setSelected(null)} title={selected.title} maxWidth="max-w-3xl">
            <div className="space-y-4">
              {selected.is_demo && (
                <div className="p-2 rounded-lg bg-warning/10 border border-warning/30 text-xs text-warning text-center">
                  This report contains demonstration data — not from a real assessment.
                </div>
              )}
              <ReportSection title="Executive Summary" content={selected.content?.executive_summary} />
              <ReportSection title="Assessment Objectives" content={selected.content?.assessment_objectives} />
              <div className="grid grid-cols-2 gap-4">
                <ReportSection title="Scope" content={(selected.content?.scope || []).join(', ')} />
                <ReportSection title="Limitations" content={(selected.content?.scope_limitations || []).join(', ') || 'None'} />
              </div>
              <ReportSection title="Methodology" content={selected.content?.methodology} />
              <div>
                <label className="text-xs text-text-secondary mb-1 block">Tools & Agents Used</label>
                <div className="flex flex-wrap gap-2">
                  {(selected.content?.tools_used || []).map((tool) => (
                    <span key={tool} className="badge bg-accent-tertiary/10 text-accent-tertiary border border-accent-tertiary/30">{tool}</span>
                  ))}
                  {(selected.content?.agents_used || []).map((agent) => (
                    <span key={agent} className="badge bg-accent-primary/10 text-accent-primary border border-accent-primary/30">{agent}</span>
                  ))}
                </div>
              </div>
              <ReportSection title="Findings Summary" content={`${selected.content?.findings_summary?.total || 0} findings identified`} />
              {selected.content?.detailed_findings?.length > 0 && (
                <div>
                  <label className="text-xs text-text-secondary mb-2 block">Detailed Findings</label>
                  <div className="space-y-2">
                    {selected.content.detailed_findings.map((f, i) => (
                      <div key={i} className="p-3 rounded-lg bg-bg-secondary border border-border">
                        <div className="flex items-center gap-2 mb-1">
                          <span className={`badge border ${f.severity === 'critical' ? 'bg-error/15 text-error border-error/30' : f.severity === 'high' ? 'bg-warning/15 text-warning border-warning/30' : 'bg-accent-secondary/15 text-accent-secondary border-accent-secondary/30'}`}>
                            {f.severity?.toUpperCase()}
                          </span>
                          <span className="text-sm text-text-primary">{f.title}</span>
                        </div>
                        <p className="text-xs text-text-secondary">{f.description}</p>
                        {f.remediation && <p className="text-xs text-accent-primary mt-1">→ {f.remediation}</p>}
                      </div>
                    ))}
                  </div>
                </div>
              )}
              <ReportSection title="Remediation Recommendations" content={selected.content?.remediation_recommendations} />
              <ReportSection title="Limitations & Uncertainties" content={selected.content?.limitations} />
              <ReportSection title="Conclusion" content={selected.content?.conclusion} />
              <div className="flex justify-end pt-2">
                <button onClick={() => exportReport(selected)} className="btn-primary text-sm flex items-center gap-2">
                  <Download className="w-3.5 h-3.5" /> Export as JSON
                </button>
              </div>
            </div>
          </Modal>
        )}
      </AnimatePresence>
    </div>
  )
}

function ReportSection({ title, content }) {
  if (!content) return null
  return (
    <div>
      <label className="text-xs text-text-secondary mb-1 block">{title}</label>
      <p className="text-sm text-text-primary bg-bg-secondary rounded-lg p-3">{content}</p>
    </div>
  )
}
