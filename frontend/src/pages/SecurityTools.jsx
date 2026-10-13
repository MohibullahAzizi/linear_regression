import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Plus, Trash2, Wrench, CheckCircle, AlertCircle } from 'lucide-react'
import { useStore } from '../store/useStore'
import { useI18n } from '../hooks/useI18n'
import { PageHeader, Card, StatusBadge, Modal } from '../components/UI'
import { toolsAPI } from '../api'

const CATEGORIES = [
  { value: 'recon', label: 'Reconnaissance', color: 'text-accent-primary' },
  { value: 'web', label: 'Web Scanning', color: 'text-accent-secondary' },
  { value: 'vuln', label: 'Vulnerability', color: 'text-accent-tertiary' },
  { value: 'crypto', label: 'Crypto/TLS', color: 'text-warning' },
  { value: 'sast', label: 'Static Analysis', color: 'text-success' },
  { value: 'container', label: 'Container', color: 'text-accent-secondary' },
  { value: 'iac', label: 'IaC', color: 'text-accent-tertiary' },
  { value: 'secrets', label: 'Secrets', color: 'text-error' },
  { value: 'exploit', label: 'Exploitation', color: 'text-error' },
]

export default function SecurityTools() {
  const { t } = useI18n()
  const tools = useStore((s) => s.tools)
  const fetchTools = useStore((s) => s.fetchTools)
  const [showAdd, setShowAdd] = useState(false)
  const [checking, setChecking] = useState(null)
  const [form, setForm] = useState({ name: '', category: 'recon', description: '', runtime_requirements: '' })

  const check = async (id) => {
    setChecking(id)
    try {
      const { data } = await toolsAPI.check(id)
      fetchTools()
      alert(`${data.status.toUpperCase()}: ${data.message}`)
    } catch (e) {
      alert('Check failed: ' + e.message)
    }
    setChecking(null)
  }

  const del = async (id) => {
    if (!confirm('Remove this tool?')) return
    await toolsAPI.delete(id)
    fetchTools()
  }

  const save = async () => {
    if (!form.name) return
    await toolsAPI.create(form)
    setShowAdd(false)
    setForm({ name: '', category: 'recon', description: '', runtime_requirements: '' })
    fetchTools()
  }

  return (
    <div>
      <PageHeader
        title={t('securityTools')}
        subtitle="Security tool catalog and integration configuration"
        action={
          <button onClick={() => setShowAdd(true)} className="btn-primary flex items-center gap-2 text-sm">
            <Plus className="w-4 h-4" /> {t('addTool')}
          </button>
        }
      />

      {/* Info banner */}
      <Card className="mb-4 !p-4 border-warning/30">
        <div className="flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-warning shrink-0 mt-0.5" />
          <div>
            <p className="text-sm text-text-primary font-medium">External Runner Required</p>
            <p className="text-xs text-text-secondary mt-1">
              Security tools require a separate execution runner service. The application cannot execute OS commands directly.
              Configure the runner in the Integrations page to enable tool execution. Tools are listed as integration targets, not installed software.
            </p>
          </div>
        </div>
      </Card>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {tools.map((tool, i) => {
          const cat = CATEGORIES.find((c) => c.value === tool.category)
          return (
            <Card key={tool.id} delay={i * 0.05}>
              <div className="flex items-start justify-between mb-3">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-lg bg-bg-secondary flex items-center justify-center">
                    <Wrench className={`w-5 h-5 ${cat?.color || 'text-text-secondary'}`} />
                  </div>
                  <div>
                    <h3 className="font-semibold text-text-primary text-sm">{tool.name}</h3>
                    <div className="flex items-center gap-2 mt-1">
                      <span className="text-[10px] text-text-secondary">{cat?.label || tool.category}</span>
                      <StatusBadge status={tool.status} />
                    </div>
                  </div>
                </div>
                {tool.is_custom && (
                  <button onClick={() => del(tool.id)} className="text-text-secondary hover:text-error p-1">
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                )}
              </div>
              <p className="text-xs text-text-secondary mb-3 line-clamp-2">{tool.description}</p>
              <div className="flex flex-wrap gap-1 mb-3">
                {tool.task_types?.slice(0, 3).map((tt) => (
                  <span key={tt} className="text-[10px] px-2 py-0.5 rounded bg-bg-secondary text-text-secondary">{tt}</span>
                ))}
              </div>
              <button
                onClick={() => check(tool.id)}
                disabled={checking === tool.id}
                className="btn-secondary text-xs w-full flex items-center justify-center gap-1"
              >
                <CheckCircle className="w-3 h-3" /> {checking === tool.id ? '...' : t('checkAvailability')}
              </button>
            </Card>
          )
        })}
      </div>

      <AnimatePresence>
        {showAdd && (
          <Modal isOpen={showAdd} onClose={() => setShowAdd(false)} title={t('addTool')}>
            <div className="space-y-4">
              <div>
                <label className="text-xs text-text-secondary mb-1 block">{t('name')}</label>
                <input className="input-field" value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} />
              </div>
              <div>
                <label className="text-xs text-text-secondary mb-1 block">{t('category')}</label>
                <select className="input-field" value={form.category} onChange={(e) => setForm({ ...form, category: e.target.value })}>
                  {CATEGORIES.map((c) => <option key={c.value} value={c.value}>{c.label}</option>)}
                </select>
              </div>
              <div>
                <label className="text-xs text-text-secondary mb-1 block">{t('description')}</label>
                <textarea className="input-field" rows="2" value={form.description} onChange={(e) => setForm({ ...form, description: e.target.value })} />
              </div>
              <div>
                <label className="text-xs text-text-secondary mb-1 block">{t('runtimeRequirements')}</label>
                <input className="input-field" value={form.runtime_requirements} onChange={(e) => setForm({ ...form, runtime_requirements: e.target.value })} />
              </div>
              <div className="flex justify-end gap-3">
                <button onClick={() => setShowAdd(false)} className="btn-secondary text-sm">{t('cancel')}</button>
                <button onClick={save} className="btn-primary text-sm">{t('create')}</button>
              </div>
            </div>
          </Modal>
        )}
      </AnimatePresence>
    </div>
  )
}
