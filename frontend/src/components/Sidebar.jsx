import { NavLink } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import {
  LayoutDashboard, Shield, Bot, Cpu, Wrench, MessageSquare,
  Workflow, Crosshair, Bug, FileText, History, ScrollText,
  Plug, Settings, ChevronLeft, Zap
} from 'lucide-react'
import { useStore } from '../store/useStore'
import { useI18n } from '../hooks/useI18n'

const navItems = [
  { to: '/', icon: LayoutDashboard, key: 'commandCenter' },
  { to: '/assessments', icon: Shield, key: 'assessments' },
  { to: '/agents', icon: Bot, key: 'aiAgents' },
  { to: '/models', icon: Cpu, key: 'aiModels' },
  { to: '/tools', icon: Wrench, key: 'securityTools' },
  { to: '/copilot', icon: MessageSquare, key: 'aiCopilot' },
  { to: '/workflows', icon: Workflow, key: 'workflowStudio' },
  { to: '/targets', icon: Crosshair, key: 'targetsScope' },
  { to: '/findings', icon: Bug, key: 'findings' },
  { to: '/reports', icon: FileText, key: 'reports' },
  { to: '/history', icon: History, key: 'executionHistory' },
  { to: '/audit', icon: ScrollText, key: 'auditLog' },
  { to: '/integrations', icon: Plug, key: 'integrations' },
  { to: '/settings', icon: Settings, key: 'settings' },
]

export default function Sidebar() {
  const collapsed = useStore((s) => s.sidebarCollapsed)
  const toggle = useStore((s) => s.toggleSidebar)
  const { t } = useI18n()

  return (
    <motion.aside
      animate={{ width: collapsed ? 72 : 256 }}
      transition={{ duration: 0.3, ease: 'easeInOut' }}
      className="bg-bg-secondary border-r border-border flex flex-col h-full relative z-20"
    >
      {/* Logo */}
      <div className="h-16 flex items-center px-4 border-b border-border gap-3">
        <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-accent-primary to-accent-secondary flex items-center justify-center shrink-0">
          <Zap className="w-5 h-5 text-bg-main" />
        </div>
        <AnimatePresence>
          {!collapsed && (
            <motion.div
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -10 }}
              className="overflow-hidden"
            >
              <div className="font-bold text-sm leading-tight text-text-primary">SENTINEL NEXUS</div>
              <div className="text-[10px] text-accent-primary font-medium">AI SECURITY PLATFORM</div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Navigation */}
      <nav className="flex-1 overflow-y-auto py-3 px-2 space-y-1">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === '/'}
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all relative group ${
                isActive
                  ? 'bg-gradient-to-r from-accent-primary/15 to-transparent text-accent-primary'
                  : 'text-text-secondary hover:text-text-primary hover:bg-bg-primary'
              }`
            }
          >
            {({ isActive }) => (
              <>
                {isActive && (
                  <motion.div
                    layoutId="sidebar-active"
                    className="absolute left-0 top-1/2 -translate-y-1/2 w-1 h-6 rounded-r bg-accent-primary"
                  />
                )}
                <item.icon className="w-5 h-5 shrink-0" />
                <AnimatePresence>
                  {!collapsed && (
                    <motion.span
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      className="text-sm font-medium whitespace-nowrap"
                    >
                      {t(item.key)}
                    </motion.span>
                  )}
                </AnimatePresence>
              </>
            )}
          </NavLink>
        ))}
      </nav>

      {/* Collapse button */}
      <button
        onClick={toggle}
        className="absolute -right-3 top-20 w-6 h-6 rounded-full bg-bg-elevated border border-border flex items-center justify-center text-text-secondary hover:text-accent-primary transition-colors z-30"
      >
        <ChevronLeft className={`w-3.5 h-3.5 transition-transform ${collapsed ? 'rotate-180' : ''}`} />
      </button>
    </motion.aside>
  )
}
