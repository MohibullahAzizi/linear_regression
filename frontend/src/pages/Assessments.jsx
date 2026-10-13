import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Plus, Trash2, Edit3, Shield, Target } from 'lucide-react'
import { useStore } from '../store/useStore'
import { useI18n } from '../hooks/useI18n'
import { PageHeader, Card, StatusBadge, Modal, EmptyState } from '../components/UI'
import { assessmentsAPI } from '../api'

const ENV_TYPES = [
  { value: 'lab', label: 'Local Laboratory' },
  { value: 'juice_shop', label: 'OWASP Juice Shop' },
  { value: 'bwa', label: 'OWASP Broken Web Apps' },
  { value: 'staging', label: 'Authorized Staging Environment' },
  { value: 'internal', label: 'Authorized Internal Assessment' },
  { value: 'custom', label: 'Custom Authorized Environment' },
]

const TARGET_TYPES = [
  { value: 'host', label: 'Host / IP' },
  { value: 'web_app', label: 'Web Application' },
  { value: 'network', label: 'Network Range' },
  { value: 'api', label: 'API Endpoint' },
]

export default function Assessments() {
  const { t } = useI18n()
  const assessments = useStore((s) => s.assessments)
  const agents = useStore((s) => s.agents)
  const models = useStore((s) => s.models)
  const tools = useStore((s) => s.tools)
  const fetchAssessments = useStore((s) => s.fetchAssessments)
  const [showForm, setShowForm] = useState(false)
  const [editing, setEditing] = useState(null)
  const [form, setForm] = useState({
    name: '', description: '', target_type: 'web_app', target_identifier: '',
    scope: [], excluded_targets: [], environment_type: 'lab',
    selected_agents: [], selected_model_id: '', selected_tools: [],
    objective: '', allowed_operations: ['reconnaissance', 'vulnerability_scanning'],
    time_limit: '', requires_approval: true, authorization_confirmed: false, notes: '',
  })
  const [scopeInput, setScopeInput] = useState('')
  const [excludedInput, setExcludedInput] = useState('')

  const openCreate = () => {
    setEditing(null)
    setForm({
      name: '', description: '', target_type: 'web_app', target_identifier: '',
      scope: [], excluded_targets: [], environment_type: 'lab',
      selected_agents: [], selected_model_id: '', selected_tools: [],
      objective: '', allowed_operations: ['reconnaissance', 'vulnerability_scanning'],
      time_limit: '', requires_approval: true, authorization_confirmed: false, notes: '',
    })
    setShowForm(true)
  }

  const openEdit = (a) => {
    setEditing(a)
    setForm({ ...a, time_limit: a.time_limit || '' })
    setShowForm(true)
  }

  const save = async () => {
    if (!form.name || !form.target_identifier) return
    const payload = { ...form, time_limit: form.time_limit ? parseInt(form.time_limit) : null }
    try {
      if (editing) {
        await assessmentsAPI.update(editing.id, payload)
      } else {
        await assessmentsAPI.create(payload)
      }
      setShowForm(false)
      fetchAssessments()
    } catch (e) {
      alert('Error saving assessment: ' + (e.response?.data?.detail || e.message))
    }
  }

  const del = async (id) => {
    if (!confirm('Delete this assessment?')) return
    await assessmentsAPI.delete(id)
    fetchAssessments()
  }

  const toggleArrayItem = (field, item) => {
    setForm((f) => ({
      ...f,
      [field]: f[field].includes(item) ? f[field].filter((x) => x !== item) : [...f[field], item],
    }))
  }

  return (
    <div>
      <PageHeader
        title={t('assessments')}
        subtitle="Manage authorized security assessments"
        action={
          <button onClick={openCreate} className="btn-primary flex items-center gap-2 text-sm">
            <Plus className="w-4 h-4" /> {t('newAssessment')}
          </button>
        }
      />

      {assessments.length === 0 ? (
        <EmptyState message="No assessments yet. Create one to get started." />
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {assessments.map((a, i) => (
            <Card key={a.id} delay={i * 0.05}>
              <div className="flex items-start justify-between mb-3">
                <div className="flex items-center gap-2">
                  {a.is_demo && <span className="badge bg-warning/15 text-warning border border-warning/30">{t('demo')}</span>}
                  <StatusBadge status={a.status} />
                </div>
                <div className="flex gap-1">
                  <button onClick={() => openEdit(a)} className="text-text-secondary hover:text-accent-primary p-1">
                    <Edit3 className="w-3.5 h-3.5" />
                  </button>
                  <button onClick={() => del(a.id)} className="text-text-secondary hover:text-error p-1">
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>
              <h3 className="font-semibold text-text-primary mb-1">{a.name}</h3>
              <p className="text-xs text-text-secondary mb-3 line-clamp-2">{a.description || a.objective}</p>
              <div className="space-y-1.5 text-xs">
                <div className="flex items-center gap-2 text-text-secondary">
                  <Target className="w-3 h-3" /> {a.target_identifier}
                </div>
                <div className="flex items-center gap-2 text-text-secondary">
                  <Shield className="w-3 h-3" /> {a.scope?.join(', ') || 'No scope defined'}
                </div>
                <div className="text-text-secondary">Progress: {a.progress}%</div>
              </div>
            </Card>
          ))}
        </div>
      )}

      <AnimatePresence>
        {showForm && (
          <Modal isOpen={showForm} onClose={() => setShowForm(false)} title={editing ? t('edit') : t('createAssessment')} maxWidth="max-w-2xl">
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-xs text-text-secondary mb-1 block">{t('name')}</label>
                  <input className="input-field" value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} placeholder="Assessment name" />
                </div>
                <div>
                  <label className="text-xs text-text-secondary mb-1 block">{t('target')}</label>
                  <input className="input-field" value={form.target_identifier} onChange={(e) => setForm({ ...form, target_identifier: e.target.value })} placeholder="e.g. http://localhost:3000" />
                </div>
              </div>

              <div>
                <label className="text-xs text-text-secondary mb-1 block">{t('description')}</label>
                <textarea className="input-field" rows="2" value={form.description} onChange={(e) => setForm({ ...form, description: e.target.value })} />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-xs text-text-secondary mb-1 block">{t('target')}</label>
                  <select className="input-field" value={form.target_type} onChange={(e) => setForm({ ...form, target_type: e.target.value })}>
                    {TARGET_TYPES.map((tt) => <option key={tt.value} value={tt.value}>{tt.label}</option>)}
                  </select>
                </div>
                <div>
                  <label className="text-xs text-text-secondary mb-1 block">{t('environmentType')}</label>
                  <select className="input-field" value={form.environment_type} onChange={(e) => setForm({ ...form, environment_type: e.target.value })}>
                    {ENV_TYPES.map((env) => <option key={env.value} value={env.value}>{env.label}</option>)}
                  </select>
                </div>
              </div>

              {/* Scope */}
              <div>
                <label className="text-xs text-text-secondary mb-1 block">{t('authorizedScope')} (e.g. localhost, 127.0.0.1, *.example.com)</label>
                <div className="flex gap-2">
                  <input className="input-field" value={scopeInput} onChange={(e) => setScopeInput(e.target.value)} placeholder="Add scope entry" onKeyDown={(e) => { if (e.key === 'Enter' && scopeInput) { setForm({ ...form, scope: [...form.scope, scopeInput] }); setScopeInput('') } }} />
                  <button onClick={() => { if (scopeInput) { setForm({ ...form, scope: [...form.scope, scopeInput] }); setScopeInput('') } }} className="btn-secondary text-sm">Add</button>
                </div>
                <div className="flex flex-wrap gap-2 mt-2">
                  {form.scope.map((s) => (
                    <span key={s} className="badge bg-accent-primary/10 text-accent-primary border border-accent-primary/30 cursor-pointer" onClick={() => setForm({ ...form, scope: form.scope.filter((x) => x !== s) })}>{s} ×</span>
                  ))}
                </div>
              </div>

              {/* Excluded */}
              <div>
                <label className="text-xs text-text-secondary mb-1 block">Excluded Targets</label>
                <div className="flex gap-2">
                  <input className="input-field" value={excludedInput} onChange={(e) => setExcludedInput(e.target.value)} placeholder="Add excluded target" onKeyDown={(e) => { if (e.key === 'Enter' && excludedInput) { setForm({ ...form, excluded_targets: [...form.excluded_targets, excludedInput] }); setExcludedInput('') } }} />
                  <button onClick={() => { if (excludedInput) { setForm({ ...form, excluded_targets: [...form.excluded_targets, excludedInput] }); setExcludedInput('') } }} className="btn-secondary text-sm">Add</button>
                </div>
                <div className="flex flex-wrap gap-2 mt-2">
                  {form.excluded_targets.map((s) => (
                    <span key={s} className="badge bg-error/10 text-error border border-error/30 cursor-pointer" onClick={() => setForm({ ...form, excluded_targets: form.excluded_targets.filter((x) => x !== s) })}>{s} ×</span>
                  ))}
                </div>
              </div>

              <div>
                <label className="text-xs text-text-secondary mb-1 block">{t('objective')}</label>
                <textarea className="input-field" rows="2" value={form.objective} onChange={(e) => setForm({ ...form, objective: e.target.value })} placeholder="e.g. Passive vulnerability assessment" />
              </div>

              {/* Agent selection */}
              <div>
                <label className="text-xs text-text-secondary mb-1 block">{t('selectedAgents')}</label>
                <div className="flex flex-wrap gap-2">
                  {agents.filter((a) => a.enabled).map((a) => (
                    <button key={a.id} onClick={() => toggleArrayItem('selected_agents', a.name)} className={`badge border transition-colors ${form.selected_agents.includes(a.name) ? 'bg-accent-primary/15 text-accent-primary border-accent-primary/30' : 'bg-bg-secondary text-text-secondary border-border'}`}>
                      {a.name}
                    </button>
                  ))}
                </div>
              </div>

              {/* Model selection */}
              <div>
                <label className="text-xs text-text-secondary mb-1 block">{t('selectedModel')}</label>
                <select className="input-field" value={form.selected_model_id} onChange={(e) => setForm({ ...form, selected_model_id: e.target.value })}>
                  <option value="">None</option>
                  {models.map((m) => <option key={m.id} value={m.id}>{m.display_name} ({m.provider})</option>)}
                </select>
              </div>

              {/* Tool selection */}
              <div>
                <label className="text-xs text-text-secondary mb-1 block">{t('selectedTools')}</label>
                <div className="flex flex-wrap gap-2">
                  {tools.filter((tool) => tool.enabled).map((tool) => (
                    <button key={tool.id} onClick={() => toggleArrayItem('selected_tools', tool.name)} className={`badge border transition-colors ${form.selected_tools.includes(tool.name) ? 'bg-accent-tertiary/15 text-accent-tertiary border-accent-tertiary/30' : 'bg-bg-secondary text-text-secondary border-border'}`}>
                      {tool.name}
                    </button>
                  ))}
                </div>
              </div>

              {/* Authorization */}
              <label className="flex items-center gap-3 cursor-pointer">
                <input type="checkbox" checked={form.authorization_confirmed} onChange={(e) => setForm({ ...form, authorization_confirmed: e.target.checked })} className="w-4 h-4 accent-accent-primary" />
                <span className="text-xs text-text-secondary">{t('authorizationConfirm')}</span>
              </label>

              <div className="flex justify-end gap-3 pt-2">
                <button onClick={() => setShowForm(false)} className="btn-secondary text-sm">{t('cancel')}</button>
                <button onClick={save} className="btn-primary text-sm">{t('save')}</button>
              </div>
            </div>
          </Modal>
        )}
      </AnimatePresence>
    </div>
  )
}
