import { useState } from 'react'
import { useStore } from '../store/useStore'
import { useI18n } from '../hooks/useI18n'
import { PageHeader, Card, StatusBadge } from '../components/UI'
import SetupGuideModal from '../components/SetupGuideModal'
import { modelsAPI, agentsAPI, toolsAPI } from '../api'
import {
  getToolGuide, getAgentGuide, getModelGuide, getRunnerGuide, STATIC_INTEGRATIONS,
} from '../lib/integrationGuides'
import {
  Cpu, Bot, Wrench, Server, Bell, Plug, Zap, AlertCircle, BookOpen, Github, MessageSquare,
} from 'lucide-react'

const STATIC_ICONS = { github: Github, slack: MessageSquare }

export default function Integrations() {
  const { t } = useI18n()
  const models = useStore((s) => s.models)
  const agents = useStore((s) => s.agents)
  const tools = useStore((s) => s.tools)
  const fetchModels = useStore((s) => s.fetchModels)
  const fetchAgents = useStore((s) => s.fetchAgents)
  const fetchTools = useStore((s) => s.fetchTools)
  const [testing, setTesting] = useState(null)
  const [guide, setGuide] = useState(null)

  const testModel = async (id) => {
    setTesting(`model-${id}`)
    await modelsAPI.test(id)
    fetchModels()
    setTesting(null)
  }
  const testAgent = async (id) => {
    setTesting(`agent-${id}`)
    await agentsAPI.test(id)
    fetchAgents()
    setTesting(null)
  }
  const testTool = async (id) => {
    setTesting(`tool-${id}`)
    await toolsAPI.check(id)
    fetchTools()
    setTesting(null)
  }

  const integrationSections = [
    {
      key: 'model',
      title: 'AI Model Providers',
      icon: Cpu,
      items: models.map((m) => ({
        id: m.id, name: m.display_name, purpose: `${m.provider} • ${m.model_identifier}`,
        status: m.status, error: m.last_error, onTest: () => testModel(m.id),
        onGuide: () => setGuide(getModelGuide(m)),
      })),
    },
    {
      key: 'agent',
      title: 'AI Agents',
      icon: Bot,
      items: agents.map((a) => ({
        id: a.id, name: a.name, purpose: a.adapter_type,
        status: a.status, error: a.last_error, onTest: () => testAgent(a.id),
        onGuide: () => setGuide(getAgentGuide(a)),
      })),
    },
    {
      key: 'tool',
      title: 'Security Tools',
      icon: Wrench,
      items: tools.map((tool) => ({
        id: tool.id, name: tool.name, purpose: tool.category,
        status: tool.status, error: tool.last_error, onTest: () => testTool(tool.id),
        onGuide: () => setGuide(getToolGuide(tool)),
      })),
    },
  ]

  return (
    <div>
      <PageHeader title={t('integrations')} subtitle="Configure external services and connections" />

      {/* Runner info */}
      <Card className="mb-4 !p-4 border-warning/30">
        <div className="flex items-start gap-3">
          <Server className="w-5 h-5 text-warning shrink-0 mt-0.5" />
          <div className="min-w-0">
            <p className="text-sm text-text-primary font-medium">External Execution Runner</p>
            <p className="text-xs text-text-secondary mt-1">
              Security tools (Nmap, Nuclei, etc.) require a separate runner service that can execute CLI commands.
              The application communicates with the runner through authenticated, validated API requests.
              Configure the runner endpoint in each tool's settings. Never expose an unrestricted shell endpoint.
            </p>
            <div className="mt-2 flex flex-wrap items-center gap-3">
              <span className="p-2 rounded-lg bg-bg-secondary text-xs text-text-secondary font-mono">
                Runner Status: Not Configured
              </span>
              <button onClick={() => setGuide(getRunnerGuide())} className="btn-secondary text-xs flex items-center gap-1">
                <BookOpen className="w-3 h-3" /> Runner Setup Guide
              </button>
            </div>
          </div>
        </div>
      </Card>

      {/* Ollama info */}
      <Card className="mb-4 !p-4 border-accent-secondary/30">
        <div className="flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-accent-secondary shrink-0 mt-0.5" />
          <div className="min-w-0">
            <p className="text-sm text-text-primary font-medium">Ollama Connection Notes</p>
            <p className="text-xs text-text-secondary mt-1">
              If the backend runs in a container or remote server, <code className="text-accent-primary">localhost:11434</code> refers to that server, not your computer.
              Options: (1) Run Ollama on the same host as the backend. (2) Use a secure tunnel (e.g. SSH port forwarding). (3) Configure a network-accessible Ollama endpoint.
              Never expose Ollama to the public internet without authentication.
            </p>
            <button
              onClick={() => setGuide(getModelGuide({ provider: 'ollama', display_name: 'Ollama' }))}
              className="btn-secondary text-xs mt-2 flex items-center gap-1"
            >
              <BookOpen className="w-3 h-3" /> Ollama Setup Guide
            </button>
          </div>
        </div>
      </Card>

      {integrationSections.map((section) => (
        <div key={section.key} className="mb-6">
          <h3 className="text-sm font-semibold text-text-primary mb-3 flex items-center gap-2">
            <section.icon className="w-4 h-4 text-accent-primary" />
            {section.title}
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {section.items.map((item) => (
              <Card key={item.id} className="!p-4">
                <div className="flex items-start justify-between mb-2">
                  <div className="flex-1 min-w-0">
                    <div className="text-sm font-medium text-text-primary truncate">{item.name}</div>
                    <div className="text-xs text-text-secondary">{item.purpose}</div>
                  </div>
                  <StatusBadge status={item.status} />
                </div>
                {item.error && (
                  <div className="text-[10px] text-error truncate mb-2">⚠ {item.error}</div>
                )}
                <div className="flex gap-2">
                  <button
                    onClick={item.onTest}
                    disabled={testing === `${section.key}-${item.id}`}
                    className="btn-secondary text-xs flex-1 flex items-center justify-center gap-1"
                  >
                    <Zap className="w-3 h-3" />
                    {testing === `${section.key}-${item.id}` ? 'Testing...' : t('testConnection')}
                  </button>
                  <button
                    onClick={item.onGuide}
                    className="btn-secondary text-xs flex items-center justify-center gap-1"
                    title="Setup Guide"
                  >
                    <BookOpen className="w-3 h-3" /> Guide
                  </button>
                </div>
              </Card>
            ))}
          </div>
        </div>
      ))}

      {/* Developer & notification integrations */}
      <div className="mb-6">
        <h3 className="text-sm font-semibold text-text-primary mb-3 flex items-center gap-2">
          <Plug className="w-4 h-4 text-accent-primary" />
          Developer & Notification Integrations
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
          {STATIC_INTEGRATIONS.map((integration) => {
            const Icon = STATIC_ICONS[integration.icon] || Bell
            return (
              <Card key={integration.id} className="!p-4">
                <div className="flex items-start justify-between mb-2">
                  <div className="flex items-start gap-3 min-w-0">
                    <div className="w-9 h-9 rounded-lg bg-bg-secondary flex items-center justify-center shrink-0">
                      <Icon className="w-4 h-4 text-accent-primary" />
                    </div>
                    <div className="min-w-0">
                      <div className="text-sm font-medium text-text-primary truncate">{integration.name}</div>
                      <div className="text-xs text-text-secondary">{integration.purpose}</div>
                    </div>
                  </div>
                  <StatusBadge status={integration.status} />
                </div>
                <button
                  onClick={() => setGuide(integration.guide)}
                  className="btn-secondary text-xs w-full flex items-center justify-center gap-1"
                >
                  <BookOpen className="w-3 h-3" /> Setup Guide
                </button>
              </Card>
            )
          })}
        </div>
      </div>

      <SetupGuideModal guide={guide} onClose={() => setGuide(null)} />
    </div>
  )
}
