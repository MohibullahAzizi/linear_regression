import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { Search, Bell, Plus, Activity, User, Globe } from 'lucide-react'
import { useStore } from '../store/useStore'
import { useI18n } from '../hooks/useI18n'

export default function TopBar() {
  const [showNotif, setShowNotif] = useState(false)
  const [showProfile, setShowProfile] = useState(false)
  const [showLang, setShowLang] = useState(false)
  const navigate = useNavigate()
  const { t, lang } = useI18n()
  const setLang = useStore((s) => s.setLang)
  const health = useStore((s) => s.health)
  const findings = useStore((s) => s.findings)
  const pendingFindings = findings.filter((f) => f.validation_status === 'new')

  return (
    <header className="h-16 bg-bg-secondary/80 backdrop-blur-md border-b border-border flex items-center justify-between px-6 shrink-0 z-10">
      {/* Search */}
      <div className="flex items-center gap-4 flex-1 max-w-md">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-text-secondary" />
          <input
            type="text"
            placeholder={t('search')}
            className="w-full bg-bg-primary border border-border rounded-lg pl-10 pr-4 py-2 text-sm text-text-primary placeholder-text-secondary focus:outline-none focus:border-accent-primary transition-colors"
          />
        </div>
      </div>

      {/* Right side */}
      <div className="flex items-center gap-3">
        {/* Health indicator */}
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-bg-primary border border-border">
          <div className={`w-2 h-2 rounded-full ${health?.status === 'ok' ? 'bg-success animate-pulse' : 'bg-error'}`} />
          <span className="text-xs text-text-secondary font-medium">
            {health?.status === 'ok' ? 'System Online' : 'System Issue'}
          </span>
        </div>

        {/* Language */}
        <div className="relative">
          <button
            onClick={() => setShowLang(!showLang)}
            className="w-9 h-9 rounded-lg bg-bg-primary border border-border flex items-center justify-center text-text-secondary hover:text-accent-primary transition-colors"
          >
            <Globe className="w-4 h-4" />
          </button>
          <AnimatePresence>
            {showLang && (
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className="absolute right-0 top-11 w-32 card-elevated rounded-lg overflow-hidden shadow-xl z-50"
              >
                <button
                  onClick={() => { setLang('en'); setShowLang(false) }}
                  className={`w-full px-4 py-2.5 text-left text-sm hover:bg-bg-primary transition-colors ${lang === 'en' ? 'text-accent-primary' : 'text-text-primary'}`}
                >
                  🇬🇧 English
                </button>
                <button
                  onClick={() => { setLang('fa'); setShowLang(false) }}
                  className={`w-full px-4 py-2.5 text-left text-sm hover:bg-bg-primary transition-colors ${lang === 'fa' ? 'text-accent-primary' : 'text-text-primary'}`}
                >
                  🇮🇷 فارسی
                </button>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Notifications */}
        <div className="relative">
          <button
            onClick={() => setShowNotif(!showNotif)}
            className="w-9 h-9 rounded-lg bg-bg-primary border border-border flex items-center justify-center text-text-secondary hover:text-accent-primary transition-colors relative"
          >
            <Bell className="w-4 h-4" />
            {pendingFindings.length > 0 && (
              <span className="absolute -top-1 -right-1 w-4 h-4 rounded-full bg-error text-[9px] flex items-center justify-center text-white font-bold">
                {pendingFindings.length}
              </span>
            )}
          </button>
          <AnimatePresence>
            {showNotif && (
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className="absolute right-0 top-11 w-80 card-elevated rounded-lg overflow-hidden shadow-xl z-50"
              >
                <div className="px-4 py-3 border-b border-border">
                  <span className="text-sm font-semibold text-text-primary">Notifications</span>
                </div>
                <div className="max-h-64 overflow-y-auto">
                  {pendingFindings.length > 0 ? (
                    pendingFindings.slice(0, 5).map((f) => (
                      <div key={f.id} className="px-4 py-2.5 border-b border-border/50 hover:bg-bg-primary cursor-pointer" onClick={() => navigate('/findings')}>
                        <div className="text-xs text-text-secondary">{f.is_demo && `[${t('demo')}] `}{f.title}</div>
                        <div className="text-[10px] text-accent-secondary mt-0.5">New finding — {f.severity}</div>
                      </div>
                    ))
                  ) : (
                    <div className="px-4 py-6 text-center text-xs text-text-secondary">No new notifications</div>
                  )}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* New Assessment */}
        <button
          onClick={() => navigate('/assessments')}
          className="btn-primary flex items-center gap-2 text-sm"
        >
          <Plus className="w-4 h-4" />
          {t('newAssessment')}
        </button>

        {/* Profile */}
        <div className="relative">
          <button
            onClick={() => setShowProfile(!showProfile)}
            className="w-9 h-9 rounded-lg bg-gradient-to-br from-accent-secondary to-accent-tertiary flex items-center justify-center text-white"
          >
            <User className="w-4 h-4" />
          </button>
          <AnimatePresence>
            {showProfile && (
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className="absolute right-0 top-11 w-48 card-elevated rounded-lg overflow-hidden shadow-xl z-50"
              >
                <div className="px-4 py-3 border-b border-border">
                  <div className="text-sm font-medium text-text-primary">Security Analyst</div>
                  <div className="text-xs text-text-secondary">analyst@sentinel.ai</div>
                </div>
                <button onClick={() => { navigate('/settings'); setShowProfile(false) }} className="w-full px-4 py-2.5 text-left text-sm text-text-secondary hover:text-accent-primary hover:bg-bg-primary transition-colors">
                  {t('settings')}
                </button>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </header>
  )
}
