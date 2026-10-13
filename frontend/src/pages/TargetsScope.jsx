import { useState } from 'react'
import { useStore } from '../store/useStore'
import { useI18n } from '../hooks/useI18n'
import { PageHeader, Card, StatusBadge, EmptyState } from '../components/UI'
import { assessmentsAPI } from '../api'
import { Target, Shield, Ban, CheckCircle, XCircle } from 'lucide-react'

export default function TargetsScope() {
  const { t } = useI18n()
  const assessments = useStore((s) => s.assessments)
  const [selected, setSelected] = useState(null)
  const [testTarget, setTestTarget] = useState('')
  const [testResult, setTestResult] = useState(null)
  const [testing, setTesting] = useState(false)

  const validateTarget = async () => {
    if (!selected || !testTarget) return
    setTesting(true)
    try {
      const { data } = await assessmentsAPI.validateScope(selected.id, testTarget)
      setTestResult(data)
    } catch (e) {
      setTestResult({ valid: false, reason: e.response?.data?.detail || e.message })
    }
    setTesting(false)
  }

  return (
    <div>
      <PageHeader title={t('targetsScope')} subtitle="Manage authorized targets and validate scope" />

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Assessment list */}
        <div className="space-y-2">
          {assessments.length === 0 ? (
            <EmptyState message="No assessments" />
          ) : assessments.map((a) => (
            <Card key={a.id} className={`cursor-pointer ${selected?.id === a.id ? 'border-accent-primary' : ''}`} >
              <div onClick={() => { setSelected(a); setTestResult(null); setTestTarget('') }}>
                <div className="flex items-center justify-between mb-1">
                  <h3 className="text-sm font-medium text-text-primary">{a.name}</h3>
                  <StatusBadge status={a.status} />
                </div>
                <p className="text-xs text-text-secondary">{a.target_identifier}</p>
              </div>
            </Card>
          ))}
        </div>

        {/* Scope details */}
        <div className="lg:col-span-2">
          {selected ? (
            <div className="space-y-4">
              <Card>
                <h3 className="text-sm font-semibold text-text-primary mb-4 flex items-center gap-2">
                  <Target className="w-4 h-4 text-accent-primary" /> Target & Scope Details
                </h3>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <label className="text-xs text-text-secondary">Target</label>
                    <p className="text-text-primary">{selected.target_identifier}</p>
                  </div>
                  <div>
                    <label className="text-xs text-text-secondary">Type</label>
                    <p className="text-text-primary">{selected.target_type}</p>
                  </div>
                  <div>
                    <label className="text-xs text-text-secondary">Environment</label>
                    <p className="text-text-primary">{selected.environment_type}</p>
                  </div>
                  <div>
                    <label className="text-xs text-text-secondary">Authorization</label>
                    <p className={selected.authorization_confirmed ? 'text-success' : 'text-error'}>
                      {selected.authorization_confirmed ? 'Confirmed' : 'Not confirmed'}
                    </p>
                  </div>
                </div>
              </Card>

              <Card>
                <h3 className="text-sm font-semibold text-text-primary mb-3 flex items-center gap-2">
                  <Shield className="w-4 h-4 text-success" /> Authorized Scope
                </h3>
                <div className="flex flex-wrap gap-2">
                  {selected.scope?.map((s) => (
                    <span key={s} className="badge bg-success/10 text-success border border-success/30">{s}</span>
                  )) || <span className="text-xs text-text-secondary">No scope defined</span>}
                </div>
              </Card>

              <Card>
                <h3 className="text-sm font-semibold text-text-primary mb-3 flex items-center gap-2">
                  <Ban className="w-4 h-4 text-error" /> Excluded Targets
                </h3>
                <div className="flex flex-wrap gap-2">
                  {selected.excluded_targets?.length ? selected.excluded_targets.map((s) => (
                    <span key={s} className="badge bg-error/10 text-error border border-error/30">{s}</span>
                  )) : <span className="text-xs text-text-secondary">No exclusions</span>}
                </div>
              </Card>

              {/* Scope validation */}
              <Card>
                <h3 className="text-sm font-semibold text-text-primary mb-3">Validate Target Against Scope</h3>
                <div className="flex gap-2">
                  <input
                    className="input-field"
                    value={testTarget}
                    onChange={(e) => setTestTarget(e.target.value)}
                    placeholder="Enter target to validate (e.g. 192.168.1.5)"
                    onKeyDown={(e) => e.key === 'Enter' && validateTarget()}
                  />
                  <button onClick={validateTarget} disabled={testing} className="btn-primary text-sm whitespace-nowrap">
                    {testing ? '...' : 'Validate'}
                  </button>
                </div>
                {testResult && (
                  <div className={`mt-3 p-3 rounded-lg ${testResult.valid ? 'bg-success/10 border border-success/30' : 'bg-error/10 border border-error/30'}`}>
                    <div className="flex items-center gap-2">
                      {testResult.valid ? <CheckCircle className="w-4 h-4 text-success" /> : <XCircle className="w-4 h-4 text-error" />}
                      <span className={`text-sm font-medium ${testResult.valid ? 'text-success' : 'text-error'}`}>
                        {testResult.valid ? 'In Scope' : 'Out of Scope'}
                      </span>
                    </div>
                    <p className="text-xs text-text-secondary mt-1">{testResult.reason}</p>
                  </div>
                )}
              </Card>
            </div>
          ) : (
            <Card className="h-full flex items-center justify-center">
              <EmptyState message="Select an assessment to view scope" />
            </Card>
          )}
        </div>
      </div>
    </div>
  )
}
