import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { History, XCircle, CheckCircle, Clock, Loader } from 'lucide-react'
import { useStore } from '../store/useStore'
import { useI18n } from '../hooks/useI18n'
import { PageHeader, Card, StatusBadge, EmptyState } from '../components/UI'
import { tasksAPI } from '../api'

export default function ExecutionHistory() {
  const { t } = useI18n()
  const fetchTasks = useStore((s) => s.fetchTasks)
  const [tasks, setTasks] = useState([])
  const [loading, setLoading] = useState(true)
  const [filter, setFilter] = useState('')

  useEffect(() => {
    tasksAPI.list().then(({ data }) => { setTasks(data); setLoading(false) }).catch(() => setLoading(false))
  }, [])

  const cancel = async (id) => {
    if (!confirm('Cancel this task?')) return
    await tasksAPI.cancel(id)
    const { data } = await tasksAPI.list()
    setTasks(data)
  }

  const filtered = filter ? tasks.filter((task) => task.status === filter) : tasks

  const statusIcons = {
    running: <Loader className="w-4 h-4 text-accent-primary animate-spin" />,
    completed: <CheckCircle className="w-4 h-4 text-success" />,
    failed: <XCircle className="w-4 h-4 text-error" />,
    queued: <Clock className="w-4 h-4 text-warning" />,
    cancelled: <XCircle className="w-4 h-4 text-text-secondary" />,
  }

  return (
    <div>
      <PageHeader title={t('executionHistory')} subtitle="Task execution records and status" />

      <Card className="mb-4 !p-3">
        <div className="flex gap-2 flex-wrap">
          {['', 'queued', 'running', 'completed', 'failed', 'cancelled'].map((s) => (
            <button
              key={s}
              onClick={() => setFilter(s)}
              className={`px-3 py-1.5 rounded-lg text-xs transition-colors ${filter === s ? 'bg-accent-primary/15 text-accent-primary border border-accent-primary/30' : 'bg-bg-secondary text-text-secondary border border-border'}`}
            >
              {s || 'All'}
            </button>
          ))}
        </div>
      </Card>

      {loading ? (
        <EmptyState message={t('loading')} />
      ) : filtered.length === 0 ? (
        <EmptyState message="No execution records" />
      ) : (
        <div className="space-y-3">
          {filtered.map((task, i) => (
            <Card key={task.id} delay={i * 0.03}>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3 flex-1 min-w-0">
                  {statusIcons[task.status] || <Clock className="w-4 h-4 text-text-secondary" />}
                  <div className="flex-1 min-w-0">
                    <div className="text-sm text-text-primary truncate">{task.description || 'Untitled task'}</div>
                    <div className="text-xs text-text-secondary mt-0.5">
                      Agent: {task.selected_agent || '—'} • Model: {task.selected_model || '—'} • Tools: {task.selected_tools?.join(', ') || '—'}
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-3 shrink-0">
                  <div className="text-right">
                    <StatusBadge status={task.status} />
                    {task.started_at && (
                      <div className="text-[10px] text-text-secondary mt-1">
                        {new Date(task.started_at).toLocaleString()}
                      </div>
                    )}
                  </div>
                  {(task.status === 'running' || task.status === 'queued') && (
                    <button onClick={() => cancel(task.id)} className="text-error text-xs hover:underline">Cancel</button>
                  )}
                </div>
              </div>
              {task.errors && (
                <div className="mt-2 p-2 rounded-lg bg-error/10 border border-error/30 text-xs text-error">{task.errors}</div>
              )}
              {task.result_summary && (
                <div className="mt-2 p-2 rounded-lg bg-bg-secondary text-xs text-text-secondary">{task.result_summary}</div>
              )}
            </Card>
          ))}
        </div>
      )}
    </div>
  )
}
