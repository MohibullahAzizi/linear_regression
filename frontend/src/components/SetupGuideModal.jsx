import { Modal } from './UI'
import { BookOpen, Terminal, KeyRound, ExternalLink, ShieldCheck } from 'lucide-react'

/**
 * Renders a single integration setup guide.
 * `guide` is the object returned by the helpers in lib/integrationGuides.js.
 * Nothing here connects anything — it is instructional content only.
 */
export default function SetupGuideModal({ guide, onClose }) {
  if (!guide) return null

  return (
    <Modal isOpen={!!guide} onClose={onClose} title={guide.title} maxWidth="max-w-2xl">
      <div className="space-y-6">
        {/* Summary */}
        <div className="flex items-start gap-3">
          <BookOpen className="w-5 h-5 text-accent-primary shrink-0 mt-0.5" />
          <p className="text-sm text-text-secondary leading-relaxed">{guide.summary}</p>
        </div>

        {/* Prerequisites */}
        {guide.prerequisites?.length > 0 && (
          <div>
            <h4 className="text-xs font-semibold text-text-primary uppercase tracking-wide mb-2">Prerequisites</h4>
            <ul className="space-y-1.5">
              {guide.prerequisites.map((p, i) => (
                <li key={i} className="flex items-start gap-2 text-sm text-text-secondary">
                  <ShieldCheck className="w-4 h-4 text-success shrink-0 mt-0.5" />
                  <span>{p}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Steps */}
        <div>
          <h4 className="text-xs font-semibold text-text-primary uppercase tracking-wide mb-3">Steps</h4>
          <ol className="space-y-4">
            {guide.steps?.map((step, i) => (
              <li key={i} className="flex gap-3">
                <span className="w-6 h-6 rounded-full bg-accent-primary/15 text-accent-primary text-xs font-semibold flex items-center justify-center shrink-0">
                  {i + 1}
                </span>
                <div className="min-w-0 flex-1">
                  <p className="text-sm font-medium text-text-primary">{step.title}</p>
                  <p className="text-xs text-text-secondary mt-1 leading-relaxed">{step.detail}</p>
                  {step.code && (
                    <pre className="mt-2 p-3 rounded-lg bg-bg-secondary border border-border text-[11px] text-text-secondary font-mono whitespace-pre-wrap break-words flex items-start gap-2">
                      <Terminal className="w-3.5 h-3.5 text-accent-primary shrink-0 mt-0.5" />
                      <code className="min-w-0">{step.code}</code>
                    </pre>
                  )}
                </div>
              </li>
            ))}
          </ol>
        </div>

        {/* Credentials */}
        {guide.credentials?.length > 0 && (
          <div>
            <h4 className="text-xs font-semibold text-text-primary uppercase tracking-wide mb-2 flex items-center gap-2">
              <KeyRound className="w-3.5 h-3.5 text-accent-secondary" /> Credentials
            </h4>
            <div className="space-y-2">
              {guide.credentials.map((c, i) => (
                <div key={i} className="p-3 rounded-lg bg-bg-secondary border border-border">
                  <div className="flex items-center justify-between gap-3">
                    <span className="text-sm font-medium text-text-primary">{c.label}</span>
                    <code className="text-[11px] text-accent-primary font-mono">{c.name}</code>
                  </div>
                  <p className="text-xs text-text-secondary mt-1">{c.where}</p>
                </div>
              ))}
            </div>
            <p className="text-[11px] text-text-secondary mt-2">
              Store credentials as secrets (environment variables) — never in source control.
            </p>
          </div>
        )}

        {/* Docs */}
        {guide.docs?.length > 0 && (
          <div className="flex flex-wrap gap-3 pt-1">
            {guide.docs.map((d, i) => (
              <a
                key={i}
                href={d.url}
                target="_blank"
                rel="noreferrer"
                className="text-xs text-accent-primary hover:underline flex items-center gap-1"
              >
                <ExternalLink className="w-3 h-3" /> {d.label}
              </a>
            ))}
          </div>
        )}
      </div>
    </Modal>
  )
}
