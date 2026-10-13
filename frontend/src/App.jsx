import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { useEffect } from 'react'
import { useStore } from './store/useStore'
import Layout from './components/Layout'
import CommandCenter from './pages/CommandCenter'
import Assessments from './pages/Assessments'
import AIAgents from './pages/AIAgents'
import AIModels from './pages/AIModels'
import SecurityTools from './pages/SecurityTools'
import AICopilot from './pages/AICopilot'
import WorkflowStudio from './pages/WorkflowStudio'
import TargetsScope from './pages/TargetsScope'
import Findings from './pages/Findings'
import Reports from './pages/Reports'
import ExecutionHistory from './pages/ExecutionHistory'
import AuditLog from './pages/AuditLog'
import Integrations from './pages/Integrations'
import Settings from './pages/Settings'

export default function App() {
  const fetchAll = useStore((s) => s.fetchAll)
  const reducedMotion = useStore((s) => s.reducedMotion)

  useEffect(() => {
    fetchAll()
    // Check reduced motion preference
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      useStore.getState().setReducedMotion(true)
    }
  }, [fetchAll])

  return (
    <div className={reducedMotion ? 'reduced-motion' : ''}>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Layout />}>
            <Route index element={<CommandCenter />} />
            <Route path="assessments" element={<Assessments />} />
            <Route path="agents" element={<AIAgents />} />
            <Route path="models" element={<AIModels />} />
            <Route path="tools" element={<SecurityTools />} />
            <Route path="copilot" element={<AICopilot />} />
            <Route path="workflows" element={<WorkflowStudio />} />
            <Route path="targets" element={<TargetsScope />} />
            <Route path="findings" element={<Findings />} />
            <Route path="reports" element={<Reports />} />
            <Route path="history" element={<ExecutionHistory />} />
            <Route path="audit" element={<AuditLog />} />
            <Route path="integrations" element={<Integrations />} />
            <Route path="settings" element={<Settings />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </div>
  )
}
