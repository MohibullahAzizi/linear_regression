import { useStore } from '../store/useStore'
import { useI18n } from '../hooks/useI18n'
import { PageHeader, Card } from '../components/UI'
import { Globe, Sparkles, Zap } from 'lucide-react'

export default function Settings() {
  const { t, lang } = useI18n()
  const setLang = useStore((s) => s.setLang)
  const reducedMotion = useStore((s) => s.reducedMotion)
  const setReducedMotion = useStore((s) => s.setReducedMotion)

  return (
    <div>
      <PageHeader title={t('settings')} subtitle="Application preferences and configuration" />

      <div className="max-w-2xl space-y-4">
        {/* Language */}
        <Card>
          <h3 className="text-sm font-semibold text-text-primary mb-4 flex items-center gap-2">
            <Globe className="w-4 h-4 text-accent-primary" />
            {t('language')}
          </h3>
          <div className="flex gap-3">
            <button
              onClick={() => setLang('en')}
              className={`flex-1 p-4 rounded-lg border transition-colors ${lang === 'en' ? 'border-accent-primary bg-accent-primary/10' : 'border-border bg-bg-secondary hover:border-accent-primary/30'}`}
            >
              <div className="text-2xl mb-2">🇬🇧</div>
              <div className="text-sm font-medium text-text-primary">English</div>
            </button>
            <button
              onClick={() => setLang('fa')}
              className={`flex-1 p-4 rounded-lg border transition-colors ${lang === 'fa' ? 'border-accent-primary bg-accent-primary/10' : 'border-border bg-bg-secondary hover:border-accent-primary/30'}`}
            >
              <div className="text-2xl mb-2">🇮🇷</div>
              <div className="text-sm font-medium text-text-primary" dir="rtl">فارسی</div>
            </button>
          </div>
        </Card>

        {/* Appearance */}
        <Card>
          <h3 className="text-sm font-semibold text-text-primary mb-4 flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-accent-primary" />
            {t('appearance')}
          </h3>
          <label className="flex items-center justify-between cursor-pointer">
            <div>
              <div className="text-sm text-text-primary">{t('reducedMotion')}</div>
              <div className="text-xs text-text-secondary mt-0.5">Minimize animations and transitions</div>
            </div>
            <button
              onClick={() => setReducedMotion(!reducedMotion)}
              className={`w-12 h-6 rounded-full transition-colors relative ${reducedMotion ? 'bg-accent-primary' : 'bg-border'}`}
            >
              <div className={`absolute top-0.5 w-5 h-5 rounded-full bg-white transition-transform ${reducedMotion ? 'translate-x-6' : 'translate-x-0.5'}`} />
            </button>
          </label>
        </Card>

        {/* About */}
        <Card>
          <h3 className="text-sm font-semibold text-text-primary mb-3 flex items-center gap-2">
            <Zap className="w-4 h-4 text-accent-primary" />
            About SENTINEL NEXUS AI
          </h3>
          <div className="space-y-2 text-xs text-text-secondary">
            <p><span className="text-text-primary font-medium">Version:</span> 1.0.0</p>
            <p><span className="text-text-primary font-medium">Tagline:</span> One Intelligence. Every Tool. Complete Control.</p>
            <p><span className="text-text-primary font-medium">Description:</span> An intelligent unified cybersecurity workspace for coordinating AI agents, security tools, and authorized penetration-testing workflows.</p>
            <p className="pt-2 text-text-secondary">
              This application is designed for authorized penetration testing and controlled laboratory environments.
              All security-sensitive operations require explicit authorization and approval.
            </p>
          </div>
        </Card>
      </div>
    </div>
  )
}
