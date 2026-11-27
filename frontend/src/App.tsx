import { useEffect, useState } from 'react'
import { predict } from './lib/api'
import { emptyDraft, toProfile, type Draft } from './lib/form'
import type { Prediction, Profile } from './lib/schema'
import { Cover } from './pages/Cover'
import { Questions } from './pages/Questions'
import { Result } from './pages/Result'

type View =
  | { name: 'cover' }
  | { name: 'questions' }
  | { name: 'result'; profile: Profile; prediction: Prediction }

export function App() {
  const [view, setView] = useState<View>({ name: 'cover' })
  // Answers live only in memory; closing the tab erases them.
  const [draft, setDraft] = useState<Draft>(emptyDraft)

  useEffect(() => { window.scrollTo(0, 0) }, [view.name])

  if (view.name === 'cover') return <Cover onOpen={() => setView({ name: 'questions' })} />
  if (view.name === 'questions') {
    return (
      <Questions
        draft={draft}
        onChange={setDraft}
        onExit={() => setView({ name: 'cover' })}
        onSubmit={async () => {
          const profile = toProfile(draft)
          setView({ name: 'result', profile, prediction: await predict(profile) })
        }}
      />
    )
  }
  return (
    <Result
      profile={view.profile}
      prediction={view.prediction}
      onEdit={() => setView({ name: 'questions' })}
      onClear={() => { setDraft(emptyDraft); setView({ name: 'cover' }) }}
    />
  )
}
