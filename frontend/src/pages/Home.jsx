import { useEffect, useState } from 'react'
import { getHealth, toApiError } from '../services/api'

/**
 * Placeholder foundation page for Phase 1.
 *
 * This just proves the frontend <-> backend wiring works (via the health
 * endpoint). The real landing page and workspace experience are built in
 * Phase 7 and will replace this component.
 */
function Home() {
  const [status, setStatus] = useState('checking')
  const [detail, setDetail] = useState('')

  useEffect(() => {
    let cancelled = false

    getHealth()
      .then((data) => {
        if (cancelled) return
        setStatus('ok')
        setDetail(`${data.app_name} · ${data.environment}`)
      })
      .catch((error) => {
        if (cancelled) return
        setStatus('error')
        setDetail(toApiError(error).message)
      })

    return () => {
      cancelled = true
    }
  }, [])

  return (
    <main className="flex min-h-screen flex-col items-center justify-center gap-4 bg-[#0b0d12] px-6 text-center">
      <h1 className="text-3xl font-semibold tracking-tight text-white">RepoCrawl</h1>
      <p className="max-w-md text-sm text-slate-400">
        Project foundation is up. The landing page and workspace UI arrive in a later phase.
      </p>
      <div className="flex items-center gap-2 rounded-full border border-slate-800 bg-slate-900/60 px-4 py-2 text-xs">
        <span
          className={
            'h-2 w-2 rounded-full ' +
            (status === 'ok'
              ? 'bg-emerald-400'
              : status === 'error'
                ? 'bg-red-400'
                : 'bg-amber-400 animate-pulse')
          }
        />
        <span className="text-slate-300">
          Backend: {status === 'checking' ? 'checking…' : detail}
        </span>
      </div>
    </main>
  )
}

export default Home
