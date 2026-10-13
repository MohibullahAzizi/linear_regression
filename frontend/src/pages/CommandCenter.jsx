import { useState, useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import {
  Shield, Bot, Cpu, Wrench, Bug, AlertCircle, Activity,
  Zap, MessageSquare, FileText, Settings, ChevronRight
} from 'lucide-react'
import { useStore } from '../store/useStore'
import { useI18n } from '../hooks/useI18n'
import AICore from '../three/AICore'
import { Card, StatusBadge } from '../components/UI'

export default function CommandCenter() {
  const { t } = useI18n()
  const navigate = useNavigate()
  const agents = useStore((s) => s.agents)
  const models = useStore((s) => s.models)
  const tools = useStore((s) => s.tools)
  const assessments = useStore((s) => s.assessments)
  const findings = useStore((s) => s.findings)
  const health = useStore((s) => s.health)
  const auditEvents = useStore((s) => s.auditEvents)
  const [selectedNode, setSelectedNode] = useState(null)

  const stats = useMemo(() => ({
    total: assessments.length,
    running: assessments.filter((a) => a.status === 'running').length,
    completed: assessments.filter((a) => a.status === 'completed').length,
    agents: agents.filter((a) => a.enabled).length,
    models: models.filter((m) => m.status === 'connected').length,
    tools: tools.filter((t) => t.enabled).length,
    pendingFindings: findings.filter((f) => f.validation_status === 'new').length,
    failures: auditEvents.filter((e) => e.event_type.includes('fail') || e.event_type.includes('error')).length,
  }), [assessments, agents, models, tools, findings, auditEvents])

  const coreState = useMemo(() => {
    if (stats.running > 0) return 'running'
    if (stats.failures > 0) return 'warning'
    if (stats.completed > 0) return 'completed'
    return 'idle'
  }, [stats])

  const nodes = useMemo(() => {
    const list = []
    agents.slice(0, 4).forEach((a) => list.push({ label: a.name, color: '#00E5CC', type: 'agent', id: a.id }))
    models.slice(0, 2).forEach((m) => list.push({ label: m.display_name, color: '#3987FF', type: 'model', id: m.id }))
    tools.slice(0, 3).forEach((tool) => list.push({ label: tool.name, color: '#9B6DFF', type: 'tool', id: tool.id }))
    return list.slice(0, 8)
  }, [agents, models, tools])

  const statCards = [
    { label: t('totalAssessments'), value: stats.total, icon: Shield, color: 'text-accent-primary' },
    { label: t('runningAssessments'), value: stats.running, icon: Activity, color: 'text-accent-secondary' },
    { label: t('completedAssessments'), value: stats.completed, icon: Shield, color: 'text-success' },
    { label: t('configuredAgents'), value: stats.agents, icon: Bot, color: 'text-accent-tertiary' },
    { label: t('connectedModels'), value: stats.models, icon: Cpu, color: 'text-accent-secondary' },
    { label: t('availableTools'), value: stats.tools, icon: Wrench, color: 'text-accent-tertiary' },
    { label: t('findingsAwaiting'), value: stats.pendingFindings, icon: Bug, color: 'text-warning' },
    { label: t('recentFailures'), value: stats.failures, icon: AlertCircle, color: 'text-error' },
  ]

  const quickActions = [
    { label: t('newAssessment'), icon: Shield, path: '/assessments', color: 'from-accent-primary to-accent-secondary' },
    { label: t('aiCopilot'), icon: MessageSquare, path: '/copilot', color: 'from-accent-secondary to-accent-tertiary' },
    { label: t('aiAgents'), icon: Bot, path: '/agents', color: 'from-accent-tertiary to-accent-primary' },
    { label: t('aiModels'), icon: Cpu, path: '/models', color: 'from-accent-secondary to-accent-primary' },
    { label: t('integrations'), icon: Settings, path: '/integrations', color: 'from-accent-tertiary to-accent-secondary' },
    { label: t('findings'), icon: Bug, path: '/findings', color: 'from-warning to-error' },
    { label: t('reports'), icon: FileText, path: '/reports', color: 'from-accent-primary to-success' },
  ]

  return (
    <div className="space-y-6">
      {/* Header */}
      <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}>
        <h1 className="text-2xl font-bold text-text-primary">{t('commandCenter')}</h1>
        <p className="text-sm text-text-secondary mt-1">One Intelligence. Every Tool. Complete Control.</p>
      </motion.div>

      {/* 3D Core + Stats */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* 3D Visualization */}
        <Card className="lg:col-span-2 h-[420px] relative overflow-hidden" delay={0.1}>
          <div className="absolute top-4 left-4 z-10">
            <div className="flex items-center gap-2">
              <Zap className="w-4 h-4 text-accent-primary" />
              <span className="text-sm font-semibold text-text-primary">AI Intelligence Core</span>
            </div>
            <p className="text-xs text-text-secondary mt-1">Click nodes to inspect • Core state: <span className="text-accent-primary capitalize">{coreState}</span></p>
          </div>
          {selectedNode !== null && nodes[selectedNode] && (
            <div className="absolute bottom-4 left-4 z-10 card-elevated px-4 py-3 rounded-lg max-w-xs">
              <div className="text-xs text-text-secondary mb-1">Selected: {nodes[selectedNode].type}</div>
              <div className="text-sm font-medium text-text-primary">{nodes[selectedNode].label}</div>
              <button
                onClick={() => navigate(`/${nodes[selectedNode].type === 'agent' ? 'agents' : nodes[selectedNode].type === 'model' ? 'models' : 'tools'}`)}
                className="text-xs text-accent-primary mt-1 flex items-center gap-1 hover:underline"
              >
                View details <ChevronRight className="w-3 h-3" />
              </button>
            </div>
          )}
          <AICore nodes={nodes} selectedNode={selectedNode} onSelectNode={setSelectedNode} coreState={coreState} />
        </Card>

        {/* System Health */}
        <Card delay={0.2}>
          <h3 className="text-sm font-semibold text-text-primary mb-4 flex items-center gap-2">
            <Activity className="w-4 h-4 text-accent-primary" />
            {t('systemHealth')}
          </h3>
          <div className="space-y-3">
            <HealthRow label="Backend" status={health?.backend === 'online' ? 'connected' : 'disconnected'} />
            <HealthRow label="Database" status={health?.database === 'connected' ? 'connected' : 'disconnected'} />
            <HealthRow label="AI Providers" status={stats.models > 0 ? 'connected' : 'not_configured'} />
            <HealthRow label="Ollama" status={models.find((m) => m.provider === 'ollama' && m.status === 'connected') ? 'connected' : 'not_configured'} />
            <HealthRow label="Agent Adapters" status={agents.filter((a) => a.status === 'connected').length > 0 ? 'connected' : 'not_configured'} />
            <HealthRow label="Tool Runtime" status="not_configured" />
          </div>
        </Card>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {statCards.map((stat, i) => (
          <Card key={i} delay={0.1 + i * 0.05} className="!p-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-2xl font-bold text-text-primary">{stat.value}</div>
                <div className="text-xs text-text-secondary mt-1">{stat.label}</div>
              </div>
              <div className={`w-10 h-10 rounded-lg bg-bg-secondary flex items-center justify-center ${stat.color}`}>
                <stat.icon className="w-5 h-5" />
              </div>
            </div>
          </Card>
        ))}
      </div>

      {/* Quick Actions + Recent Activity */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card delay={0.3}>
          <h3 className="text-sm font-semibold text-text-primary mb-4">{t('quickActions')}</h3>
          <div className="grid grid-cols-2 gap-3">
            {quickActions.map((action, i) => (
              <motion.button
                key={i}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => navigate(action.path)}
                className={`flex items-center gap-3 p-3 rounded-lg bg-gradient-to-br ${action.color} bg-opacity-10 border border-border hover:border-transparent transition-all text-left`}
              >
                <div className="w-8 h-8 rounded-lg bg-bg-main/40 flex items-center justify-center shrink-0">
                  <action.icon className="w-4 h-4 text-white" />
                </div>
                <span className="text-xs font-medium text-white">{action.label}</span>
              </motion.button>
            ))}
          </div>
        </Card>

        <Card delay={0.35}>
          <h3 className="text-sm font-semibold text-text-primary mb-4">{t('recentActivity')}</h3>
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {auditEvents.length > 0 ? auditEvents.slice(0, 8).map((event, i) => (
              <div key={event.id} className="flex items-start gap-3 py-2 border-b border-border/40 last:border-0">
                <div className="w-2 h-2 rounded-full bg-accent-primary mt-1.5 shrink-0" />
                <div className="flex-1 min-w-0">
                  <div className="text-xs text-text-primary truncate">{event.description}</div>
                  <div className="text-[10px] text-text-secondary mt-0.5">
                    {new Date(event.created_at).toLocaleString()} • {event.event_type}
                  </div>
                </div>
              </div>
            )) : (
              <div className="text-xs text-text-secondary text-center py-8">{t('noData')}</div>
            )}
          </div>
        </Card>
      </div>
    </div>
  )
}

function HealthRow({ label, status }) {
  const colors = {
    connected: 'bg-success',
    not_configured: 'bg-text-secondary',
    disconnected: 'bg-error',
  }
  return (
    <div className="flex items-center justify-between">
      <span className="text-xs text-text-secondary">{label}</span>
      <div className="flex items-center gap-2">
        <div className={`w-2 h-2 rounded-full ${colors[status] || colors.not_configured}`} />
        <StatusBadge status={status} />
      </div>
    </div>
  )
}
