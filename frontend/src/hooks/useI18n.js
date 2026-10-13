import { useStore } from '../store/useStore'
import { translations } from '../lib/i18n'

export function useI18n() {
  const lang = useStore((s) => s.lang)
  const t = (key) => translations[lang]?.[key] || translations.en[key] || key
  return { t, lang }
}
