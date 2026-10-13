import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Plus, Trash2, Workflow, Copy, Save, Play, ArrowRight } from 'lucide-react'
import { useStore } from '../store/useStore'
import { useI18n } from '../hooks/useI18n'
import { PageHeader, Card, StatusBadge, Modal, EmptyState } from '../components/UI'
import { workflowsAPI } from '../api'

const STEP_TYPES = [
  'start', 'validate_scope', 'check_integration', 'select_agent', 'select_model',
  'run_tool', 'collect_output', 'analyze', 'normalize', 'request_approval',
  'generate_report', 'complete',
]

export default function WorkflowStudio() {
  const { t } = useI18n()
  const workflows = useStore((s) => s.workflows)
  const fetchWorkflows = useStore((s) => s.fetchWorkflows)
  const [selected, setSelected] = useState(null)
  const [showAdd, setShowAdd] = useState(false)
  const [form, setForm] = useState({ name: '', description: '', steps: [] })

  const create = async () => {
    if (!form.name) return
    const { data } = await workflowsAPI.create({ ...form, steps: form.steps.length ? form.steps : [{ id: 's1', name: 'Start', type: 'start', status: 'pending', config: {} }] })
    setShowAdd(false)
    setForm({ name: '', description: '', steps: [] })
    fetchWorkflows()
    setSelected(data)
  }

  const duplicate = async (id) => {
    await workflowsAPI.duplicate(id)
    fetchWorkflows()
  }

  const del = async (id) => {
    if (!confirm('Delete this workflow?')) return
    await workflowsAPI.delete(id)
    setSelected(null)
    fetchWorkflows()
  }

  const saveSteps = async () => {
    if (!selected) return
    await workflowsAPI.update(selected.id, { steps: selected.steps })
    fetchWorkflows()
  }

  const addStep = () => {
    const newStep = { id: `s${Date.now()}`, name: 'New Step', type: 'run_tool', status: 'pending', config: {} }
    setSelected({ ...selected, steps: [...(selected.steps || []), newStep] })
  }

  const updateStep = (idx, field, value) => {
    const steps = [...selected.steps]
    steps[idx] = { ...steps[idx], [field]: value }
    setSelected({ ...selected, steps })
  }

  const removeStep = (idx) => {
    setSelected({ ...selected, steps: selected.steps.filter((_, i) => i !== idx) })
  }

  return (
    <div>
      <PageHeader
        title={t('workflowStudio')}
        subtitle="Create and edit assessment workflows"
        action={
          <button onClick={() => setShowAdd(true)} className="btn-primary flex items-center gap-2 text-sm">
            <Plus className="w-4 h-4" /> Create Workflow
          </button>
        }
      />

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Workflow list */}
        <div className="space-y-2">
          {workflows.length === 0 ? (
            <EmptyState message="No workflows yet" />
          ) : workflows.map((w) => (
            <Card key={w.id} className={`cursor-pointer ${selected?.id === w.id ? 'border-accent-primary' : ''}`} >
              <div onClick={() => setSelected(w)}>
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <Workflow className="w-4 h-4 text-accent-primary" />
                    <h3 className="text-sm font-medium text-text-primary">{w.name}</h3>
                  </div>
                  <div className="flex gap-1">
                    <button onClick={(e) => { e.stopPropagation(); duplicate(w.id) }} className="text-text-secondary hover:text-accent-primary p-1">
                      <Copy className="w-3 h-3" />
                    </button>
                    <button onClick={(e) => { e.stopPropagation(); del(w.id) }} className="text-text-secondary hover:text-error p-1">
                      <Trash2 className="w-3 h-3" />
                    </button>
                  </div>
                </div>
                <p className="text-xs text-text-secondary line-clamp-1">{w.description}</p>
                <div className="text-[10px] text-text-secondary mt-1">{w.steps?.length || 0} steps</div>
              </div>
            </Card>
          ))}
        </div>

        {/* Workflow editor */}
        <div className="lg:col-span-2">
          {selected ? (
            <Card>
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-semibold text-text-primary">{selected.name}</h2>
                <div className="flex gap-2">
                  <button onClick={addStep} className="btn-secondary text-xs flex items-center gap-1">
                    <Plus className="w-3 h-3" /> Add Step
                  </button>
                  <button onClick={saveSteps} className="btn-primary text-xs flex items-center gap-1">
                    <Save className="w-3 h-3" /> Save
                  </button>
                </div>
              </div>

              {/* Steps */}
              <div className="space-y-2">
                {(selected.steps || []).map((step, idx) => (
                  <motion.div
                    key={step.id}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    className="flex items-center gap-2 p-3 rounded-lg bg-bg-secondary border border-border"
                  >
                    <span className="w-6 h-6 rounded-full bg-accent-primary/20 text-accent-primary text-xs flex items-center justify-center font-bold shrink-0">{idx + 1}</span>
                    <input
                      value={step.name}
                      onChange={(e) => updateStep(idx, 'name', e.target.value)}
                      className="input-field text-sm flex-1"
                    />
                    <select
                      value={step.type}
                      onChange={(e) => updateStep(idx, 'type', e.target.value)}
                      className="input-field text-xs w-40"
                    >
                      {STEP_TYPES.map((st) => <option key={st} value={st}>{st}</option>)}
                    </select>
                    <button onClick={() => removeStep(idx)} className="text-text-secondary hover:text-error p-1 shrink-0">
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                    {idx < (selected.steps || []).length - 1 && (
                      <ArrowRight className="w-3 h-3 text-text-secondary absolute" style={{ left: '50%', bottom: '-12px' }} />
                    )}
                  </motion.div>
                ))}
              </div>

              <div className="mt-4 p-3 rounded-lg bg-warning/10 border border-warning/30">
                <p className="text-xs text-warning">
                  ⚠ Workflows can only run when all required integrations are configured. Steps requiring unavailable tools or agents will show an error.
                </p>
              </div>
            </Card>
          ) : (
            <Card className="h-full flex items-center justify-center">
              <EmptyState message="Select a workflow to edit" />
            </Card>
          )}
        </div>
      </div>

      <AnimatePresence>
        {showAdd && (
          <Modal isOpen={showAdd} onClose={() => setShowAdd(false)} title="Create Workflow">
            <div className="space-y-4">
              <div>
                <label className="text-xs text-text-secondary mb-1 block">{t('name')}</label>
                <input className="input-field" value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} />
              </div>
              <div>
                <label className="text-xs text-text-secondary mb-1 block">{t('description')}</label>
                <textarea className="input-field" rows="2" value={form.description} onChange={(e) => setForm({ ...form, description: e.target.value })} />
              </div>
              <div className="flex justify-end gap-3">
                <button onClick={() => setShowAdd(false)} className="btn-secondary text-sm">{t('cancel')}</button>
                <button onClick={create} className="btn-primary text-sm">{t('create')}</button>
              </div>
            </div>
          </Modal>
        )}
      </AnimatePresence>
    </div>
  )
}
