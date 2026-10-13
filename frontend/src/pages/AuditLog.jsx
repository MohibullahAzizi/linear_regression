import { useStore } from '../store/useStore'
import { useI18n } from '../hooks/useI18n'
import { PageHeader, Card, EmptyState } from '../components/UI'
import { ScrollText } from 'lucide-react'

export default function AuditLog() {
  const { t } = useI18n()
  const auditEvents = useStore((s) => s.auditEvents)

  return (
    <div>
      <PageHeader title={t('auditLog')} subtitle="Security and configuration event records" />

      {auditEvents.length === 0 ? (
        <EmptyState message="No audit events recorded" />
      ) : (
        <Card>
          <div className="space-y-1">
            {auditEvents.map((event, i) => (
              <div key={event.id} className="flex items-start gap-3 py-2.5 border-b border-border/40 last:border-0">
                <div className="w-8 h-8 rounded-lg bg-bg-secondary flex items-center justify-center shrink-0">
                  <ScrollText className="w-4 h-4 text-accent-secondary" />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="text-sm text-text-primary">{event.description}</div>
                  <div className="text-xs text-text-secondary mt-0.5">
                    <span className="font-mono">{event.event_type}</span>
                    {' • '}
                    {new Date(event.created_at).toLocaleString()}
                    {event.entity_type && ` • ${event.entity_type}`}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  )
}
