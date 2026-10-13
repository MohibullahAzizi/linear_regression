import { motion } from 'framer-motion'
import { Loader2 } from 'lucide-react'

export function LoadingSpinner({ size = 24 }) {
  return <Loader2 className="animate-spin text-accent-primary" style={{ width: size, height: size }} />
}

export function PageLoader() {
  return (
    <div className="flex items-center justify-center h-64">
      <LoadingSpinner size={32} />
    </div>
  )
}

export function PageHeader({ title, subtitle, action }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      className="flex items-center justify-between mb-6"
    >
      <div>
        <h1 className="text-2xl font-bold text-text-primary">{title}</h1>
        {subtitle && <p className="text-sm text-text-secondary mt-1">{subtitle}</p>}
      </div>
      {action}
    </motion.div>
  )
}

export function StatusBadge({ status }) {
  const colors = {
    connected: 'bg-success/15 text-success border-success/30',
    configured: 'bg-accent-secondary/15 text-accent-secondary border-accent-secondary/30',
    not_configured: 'bg-text-secondary/10 text-text-secondary border-border',
    disconnected: 'bg-error/15 text-error border-error/30',
    running: 'bg-accent-primary/15 text-accent-primary border-accent-primary/30',
    completed: 'bg-success/15 text-success border-success/30',
    failed: 'bg-error/15 text-error border-error/30',
    cancelled: 'bg-text-secondary/15 text-text-secondary border-border',
    queued: 'bg-warning/15 text-warning border-warning/30',
    waiting_approval: 'bg-warning/15 text-warning border-warning/30',
    available: 'bg-success/15 text-success border-success/30',
    unavailable: 'bg-error/15 text-error border-error/30',
    disabled: 'bg-text-secondary/10 text-text-secondary border-border',
    active: 'bg-success/15 text-success border-success/30',
    draft: 'bg-text-secondary/10 text-text-secondary border-border',
    paused: 'bg-warning/15 text-warning border-warning/30',
  }
  const labels = {
    connected: 'Connected',
    configured: 'Configured',
    not_configured: 'Not Configured',
    disconnected: 'Disconnected',
    running: 'Running',
    completed: 'Completed',
    failed: 'Failed',
    cancelled: 'Cancelled',
    queued: 'Queued',
    waiting_approval: 'Waiting Approval',
    available: 'Available',
    unavailable: 'Unavailable',
    disabled: 'Disabled',
    active: 'Active',
    draft: 'Draft',
    paused: 'Paused',
  }
  return (
    <span className={`badge border ${colors[status] || colors.not_configured}`}>
      {labels[status] || status}
    </span>
  )
}

export function SeverityBadge({ severity }) {
  const colors = {
    critical: 'bg-error/15 text-error border-error/30',
    high: 'bg-warning/15 text-warning border-warning/30',
    medium: 'bg-accent-secondary/15 text-accent-secondary border-accent-secondary/30',
    low: 'bg-accent-primary/15 text-accent-primary border-accent-primary/30',
    informational: 'bg-text-secondary/10 text-text-secondary border-border',
    unknown: 'bg-text-secondary/10 text-text-secondary border-border',
  }
  return (
    <span className={`badge border ${colors[severity] || colors.informational}`}>
      {severity?.toUpperCase()}
    </span>
  )
}

export function EmptyState({ message }) {
  return (
    <div className="flex flex-col items-center justify-center py-16 text-text-secondary">
      <p className="text-sm">{message}</p>
    </div>
  )
}

export function Card({ children, className = '', delay = 0 }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.3 }}
      className={`card p-5 ${className}`}
    >
      {children}
    </motion.div>
  )
}

export function Modal({ isOpen, onClose, title, children, maxWidth = 'max-w-lg' }) {
  if (!isOpen) return null
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose} />
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.95 }}
        className={`relative card-elevated w-full ${maxWidth} max-h-[85vh] overflow-y-auto rounded-xl shadow-2xl`}
      >
        <div className="flex items-center justify-between px-6 py-4 border-b border-border sticky top-0 bg-bg-elevated z-10">
          <h2 className="text-lg font-semibold text-text-primary">{title}</h2>
          <button onClick={onClose} className="text-text-secondary hover:text-text-primary text-xl">×</button>
        </div>
        <div className="p-6">{children}</div>
      </motion.div>
    </div>
  )
}
