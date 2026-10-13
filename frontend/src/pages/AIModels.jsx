import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Plus, Trash2, Cpu, Zap, Star, CheckCircle } from 'lucide-react'
import { useStore } from '../store/useStore'
import { useI18n } from '../hooks/useI18n'
import { PageHeader, Card, StatusBadge, Modal } from '../components/UI'
import { modelsAPI } from '../api'

const PROVIDERS = [
  { value: 'ollama', label: 'Ollama (Local)' },
  { value: 'openai', label: 'OpenAI' },
  { value: 'custom', label: 'OpenAI-compatible / Custom' },
]

export default function AIModels() {
  const { t } = useI18n()
  const models = useStore((s) => s.models)
  const fetchModels = useStore((s) => s.fetchModels)
  const [showAdd, setShowAdd] = useState(false)
  const [testing, setTesting] = useState(null)
  const [form, setForm] = useState({ display_name: '', provider: 'ollama', model_identifier: 'llama3.2', endpoint: 'http://localhost:11434', notes: '' })

  const test = async (id) => {
    setTesting(id)
    try {
      const { data } = await modelsAPI.test(id)
      fetchModels()
      alert(`${data.status.toUpperCase()}: ${data.message}`)
    } catch (e) {
      alert('Test failed: ' + e.message)
    }
    setTesting(null)
  }

  const activate = async (id) => {
    await modelsAPI.activate(id)
    fetchModels()
  }

  const del = async (id) => {
    if (!confirm('Remove this model?')) return
    await modelsAPI.delete(id)
    fetchModels()
  }

  const save = async () => {
    if (!form.display_name) return
    await modelsAPI.create(form)
    setShowAdd(false)
    setForm({ display_name: '', provider: 'ollama', model_identifier: 'llama3.2', endpoint: 'http://localhost:11434', notes: '' })
    fetchModels()
  }

  return (
    <div>
      <PageHeader
        title={t('aiModels')}
        subtitle="Configure AI model providers and select active models"
        action={
          <button onClick={() => setShowAdd(true)} className="btn-primary flex items-center gap-2 text-sm">
            <Plus className="w-4 h-4" /> {t('addModel')}
          </button>
        }
      />

      {/* Ollama info banner */}
      <Card className="mb-4 !p-4 border-accent-secondary/30">
        <div className="flex items-start gap-3">
          <Cpu className="w-5 h-5 text-accent-secondary shrink-0 mt-0.5" />
          <div>
            <p className="text-sm text-text-primary font-medium">Ollama Local Model Configuration</p>
            <p className="text-xs text-text-secondary mt-1">
              Ollama runs on your local machine. If the backend is hosted remotely, <code className="text-accent-primary">localhost</code> refers to the server, not your computer.
              Use a secure tunnel or configure the backend to reach your Ollama instance. Suggested model: <code className="text-accent-primary">llama3.2</code>.
              Never expose Ollama to the public internet without authentication.
            </p>
          </div>
        </div>
      </Card>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {models.map((m, i) => (
          <Card key={m.id} delay={i * 0.05}>
            <div className="flex items-start justify-between mb-3">
              <div className="flex items-center gap-3">
                <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${m.is_active ? 'bg-accent-primary/20' : 'bg-bg-secondary'}`}>
                  {m.is_active ? <CheckCircle className="w-5 h-5 text-accent-primary" /> : <Cpu className="w-5 h-5 text-text-secondary" />}
                </div>
                <div>
                  <h3 className="font-semibold text-text-primary text-sm">{m.display_name}</h3>
                  <div className="flex items-center gap-2 mt-1">
                    <StatusBadge status={m.status} />
                    {m.is_default && <span className="badge bg-warning/15 text-warning border border-warning/30"><Star className="w-2.5 h-2.5" /> Default</span>}
                  </div>
                </div>
              </div>
            </div>
            <div className="space-y-1 text-xs text-text-secondary mb-3">
              <div>Provider: <span className="text-text-primary">{m.provider}</span></div>
              <div>Model: <span className="text-text-primary font-mono">{m.model_identifier}</span></div>
              <div>Endpoint: <span className="text-text-primary font-mono text-[10px]">{m.endpoint || '—'}</span></div>
            </div>
            {m.last_error && (
              <div className="text-[10px] text-error mb-3 truncate">⚠ {m.last_error}</div>
            )}
            <div className="flex gap-2">
              <button onClick={() => test(m.id)} disabled={testing === m.id} className="btn-secondary text-xs flex-1 flex items-center justify-center gap-1">
                <Zap className="w-3 h-3" /> {testing === m.id ? '...' : t('test')}
              </button>
              {!m.is_active && (
                <button onClick={() => activate(m.id)} className="btn-primary text-xs flex-1">{t('activate')}</button>
              )}
              <button onClick={() => del(m.id)} className="text-text-secondary hover:text-error p-1.5">
                <Trash2 className="w-3.5 h-3.5" />
              </button>
            </div>
          </Card>
        ))}
      </div>

      <AnimatePresence>
        {showAdd && (
          <Modal isOpen={showAdd} onClose={() => setShowAdd(false)} title={t('addModel')}>
            <div className="space-y-4">
              <div>
                <label className="text-xs text-text-secondary mb-1 block">Display Name</label>
                <input className="input-field" value={form.display_name} onChange={(e) => setForm({ ...form, display_name: e.target.value })} placeholder="e.g. Ollama llama3.2" />
              </div>
              <div>
                <label className="text-xs text-text-secondary mb-1 block">{t('provider')}</label>
                <select className="input-field" value={form.provider} onChange={(e) => setForm({ ...form, provider: e.target.value })}>
                  {PROVIDERS.map((p) => <option key={p.value} value={p.value}>{p.label}</option>)}
                </select>
              </div>
              <div>
                <label className="text-xs text-text-secondary mb-1 block">{t('modelId')}</label>
                <input className="input-field" value={form.model_identifier} onChange={(e) => setForm({ ...form, model_identifier: e.target.value })} placeholder="llama3.2" />
              </div>
              <div>
                <label className="text-xs text-text-secondary mb-1 block">{t('endpoint')}</label>
                <input className="input-field" value={form.endpoint} onChange={(e) => setForm({ ...form, endpoint: e.target.value })} placeholder="http://localhost:11434" />
              </div>
              <div>
                <label className="text-xs text-text-secondary mb-1 block">Notes</label>
                <textarea className="input-field" rows="2" value={form.notes} onChange={(e) => setForm({ ...form, notes: e.target.value })} />
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
