import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Plus, Trash2, Bot, Plug, X } from 'lucide-react'
import { useStore } from '../store/useStore'
import { useI18n } from '../hooks/useI18n'
import { PageHeader, Card, StatusBadge, Modal, EmptyState } from '../components/UI'
import { agentsAPI } from '../api'

export default function AIAgents() {
  const { t } = useI18n()
  const agents = useStore((s) => s.agents)
  const fetchAgents = useStore((s) => s.fetchAgents)
  const [selected, setSelected] = useState(null)
  const [showAdd, setShowAdd] = useState(false)
  const [testing, setTesting] = useState(null)
  const [form, setForm] = useState({ name: '', description: '', adapter_type: 'custom', endpoint: '', capabilities: [], runtime_requirements: '' })

  const testConnection = async (id) => {
    setTesting(id)
    try {
      const { data } = await agentsAPI.test(id)
      fetchAgents()
      alert(`${data.status.toUpperCase()}: ${data.message}`)
    } catch (e) {
      alert('Test failed: ' + e.message)
    }
    setTesting(null)
  }

  const del = async (id) => {
    if (!confirm('Remove this agent?')) return
    await agentsAPI.delete(id)
    fetchAgents()
    setSelected(null)
  }

  const save = async () => {
    if (!form.name) return
    await agentsAPI.create(form)
    setShowAdd(false)
    setForm({ name: '', description: '', adapter_type: 'custom', endpoint: '', capabilities: [], runtime_requirements: '' })
    fetchAgents()
  }

  return (
    <div>
      <PageHeader
        title={t('aiAgents')}
        subtitle="Browse, configure, and manage AI agents"
        action={
          <button onClick={() => setShowAdd(true)} className="btn-primary flex items-center gap-2 text-sm">
            <Plus className="w-4 h-4" /> {t('addAgent')}
          </button>
        }
      />

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {agents.map((agent, i) => (
          <Card key={agent.id} delay={i * 0.05}>
            <div className="flex items-start justify-between mb-3">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-accent-primary/20 to-accent-secondary/20 flex items-center justify-center">
                  <Bot className="w-5 h-5 text-accent-primary" />
                </div>
                <div>
                  <h3 className="font-semibold text-text-primary text-sm">{agent.name}</h3>
                  <StatusBadge status={agent.status} />
                </div>
              </div>
            </div>
            <p className="text-xs text-text-secondary mb-3 line-clamp-2">{agent.description}</p>
            <div className="flex flex-wrap gap-1 mb-3">
              {agent.capabilities?.slice(0, 3).map((c) => (
                <span key={c} className="text-[10px] px-2 py-0.5 rounded bg-bg-secondary text-text-secondary">{c}</span>
              ))}
            </div>
            <div className="flex gap-2">
              <button onClick={() => setSelected(agent)} className="btn-secondary text-xs flex-1">Details</button>
              <button
                onClick={() => testConnection(agent.id)}
                disabled={testing === agent.id}
                className="btn-primary text-xs flex-1 flex items-center justify-center gap-1"
              >
                <Plug className="w-3 h-3" /> {testing === agent.id ? '...' : t('testConnection')}
              </button>
            </div>
          </Card>
        ))}
      </div>

      {/* Detail Modal */}
      <AnimatePresence>
        {selected && (
          <Modal isOpen={!!selected} onClose={() => setSelected(null)} title={selected.name} maxWidth="max-w-xl">
            <div className="space-y-4">
              <p className="text-sm text-text-secondary">{selected.description}</p>
              <div className="grid grid-cols-2 gap-4">
                <InfoRow label={t('integrationType')} value={selected.adapter_type} />
                <InfoRow label={t('status')} value={<StatusBadge status={selected.status} />} />
                <InfoRow label="Endpoint" value={selected.endpoint || 'Not configured'} />
                <InfoRow label={t('lastHealthCheck')} value={selected.last_health_check ? new Date(selected.last_health_check).toLocaleString() : 'Never'} />
              </div>
              <div>
                <label className="text-xs text-text-secondary mb-1 block">{t('capabilities')}</label>
                <div className="flex flex-wrap gap-2">
                  {selected.capabilities?.map((c) => (
                    <span key={c} className="badge bg-accent-primary/10 text-accent-primary border border-accent-primary/30">{c}</span>
                  ))}
                </div>
              </div>
              <div>
                <label className="text-xs text-text-secondary mb-1 block">{t('runtimeRequirements')}</label>
                <p className="text-sm text-text-primary bg-bg-secondary rounded-lg p-3">{selected.runtime_requirements}</p>
              </div>
              <div>
                <label className="text-xs text-text-secondary mb-1 block">{t('supportedModels')}</label>
                <div className="flex flex-wrap gap-2">
                  {selected.supported_models?.map((m) => (
                    <span key={m} className="badge bg-accent-secondary/10 text-accent-secondary border border-accent-secondary/30">{m}</span>
                  ))}
                </div>
              </div>
              {selected.last_error && (
                <div className="p-3 rounded-lg bg-error/10 border border-error/30">
                  <label className="text-xs text-error mb-1 block">{t('lastError')}</label>
                  <p className="text-xs text-text-primary">{selected.last_error}</p>
                </div>
              )}
              <div className="flex justify-between pt-2">
                {selected.is_custom && (
                  <button onClick={() => del(selected.id)} className="text-error text-sm flex items-center gap-1 hover:underline">
                    <Trash2 className="w-3.5 h-3.5" /> Remove
                  </button>
                )}
                <button onClick={() => testConnection(selected.id)} className="btn-primary text-sm ml-auto">
                  <Plug className="w-3.5 h-3.5" /> {t('testConnection')}
                </button>
              </div>
            </div>
          </Modal>
        )}
      </AnimatePresence>

      {/* Add Agent Modal */}
      <AnimatePresence>
        {showAdd && (
          <Modal isOpen={showAdd} onClose={() => setShowAdd(false)} title={t('addAgent')}>
            <div className="space-y-4">
              <div>
                <label className="text-xs text-text-secondary mb-1 block">{t('name')}</label>
                <input className="input-field" value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} />
              </div>
              <div>
                <label className="text-xs text-text-secondary mb-1 block">{t('description')}</label>
                <textarea className="input-field" rows="2" value={form.description} onChange={(e) => setForm({ ...form, description: e.target.value })} />
              </div>
              <div>
                <label className="text-xs text-text-secondary mb-1 block">{t('endpoint')}</label>
                <input className="input-field" value={form.endpoint} onChange={(e) => setForm({ ...form, endpoint: e.target.value })} placeholder="http://localhost:8080" />
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

function InfoRow({ label, value }) {
  return (
    <div>
      <label className="text-xs text-text-secondary mb-1 block">{label}</label>
      <div className="text-sm text-text-primary">{value}</div>
    </div>
  )
}
