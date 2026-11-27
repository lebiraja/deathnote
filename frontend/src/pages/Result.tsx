import { motion } from 'motion/react'
import { useState, type ReactNode } from 'react'
import { Factors } from '../components/Factors'
import { Lifeline } from '../components/Lifeline'
import { SurvivalChart } from '../components/SurvivalChart'
import { Button } from '../components/ui/button'
import { WhatIf } from '../components/WhatIf'
import { report } from '../lib/api'
import type { Prediction, Profile } from '../lib/schema'

function Page({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="sheet py-8 pl-14 pr-5 sm:pr-10">
      <h2 className="mb-5 font-display text-4xl font-bold">{title}</h2>
      {children}
    </section>
  )
}

interface Props {
  profile: Profile
  prediction: Prediction
  onEdit: () => void
  onClear: () => void
}

export function Result({ profile, prediction, onEdit, onClear }: Props) {
  const [failure, setFailure] = useState('')
  const [busy, setBusy] = useState(false)
  const { estimate, average } = prediction

  const download = async () => {
    setBusy(true)
    setFailure('')
    try {
      const url = URL.createObjectURL(await report(profile))
      const link = Object.assign(document.createElement('a'), { href: url, download: 'life-estimate.pdf' })
      link.click()
      setTimeout(() => URL.revokeObjectURL(url), 1000)
    } catch (error) {
      setFailure(error instanceof Error ? error.message : 'Could not make the report.')
    } finally {
      setBusy(false)
    }
  }

  return (
    <motion.main initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="mx-auto max-w-3xl space-y-8 px-4 py-8">
      <p className="font-display text-2xl font-bold">death<span className="text-blood">note</span></p>

      <section className="sheet py-8 pl-14 pr-5 sm:pr-10">
        <h1 className="font-display text-3xl">Your middle estimate</h1>
        <p className="font-hand text-8xl leading-none" data-testid="median-age">{Math.round(estimate.median_age)}</p>
        <p className="mt-3 max-w-lg text-lg">
          Half of people with your answers live between <strong>{Math.round(estimate.lower_age)}</strong> and{' '}
          <strong>{Math.round(estimate.upper_age)}</strong>. One in four live longer than that, and one in four less.
        </p>
        <p className="mt-1 text-faded">
          An average Indian {profile.sex === 'male' ? 'man' : 'woman'} aged {profile.age}: {Math.round(average.median_age)}.
        </p>
        <div className="mt-8"><Lifeline age={profile.age} estimate={estimate} average={average} /></div>
      </section>

      <Page title="Chance of being alive">
        <SurvivalChart curve={prediction.curve} labels={{ you: 'You', average: 'Average Indian' }} />
      </Page>

      <Page title="What moves your number">
        <p className="mb-5 text-sm text-faded">
          Compared with someone who sleeps 7.5 hours, never used tobacco or alcohol, exercises 300 minutes a week,
          eats very well, has a BMI of 22 and no listed illness. Yours: {prediction.sleep_hours} hours of sleep,
          BMI {prediction.bmi}.
        </p>
        <Factors factors={prediction.factors} />
      </Page>

      <Page title="Rewrite your habits"><WhatIf profile={profile} /></Page>

      {prediction.advice.length > 0 && (
        <Page title="Worth doing">
          <p className="mb-4 text-sm text-faded">General health guidance, separate from the calculation above.</p>
          <ul className="space-y-4">
            {prediction.advice.map((item) => (
              <li key={item.title}>
                <h3 className="font-display text-2xl font-bold">{item.title}</h3>
                <p>{item.detail}</p>
              </li>
            ))}
          </ul>
        </Page>
      )}

      <Page title="How far to trust this">
        <ul className="list-disc space-y-2 pl-5">
          <li>Death rates by age come from India's official life table for 2020–24, which includes the COVID-19 years.</li>
          <li>The effect of each habit was measured in US adults. It may differ for Indians.</li>
          <li>Chewing tobacco uses a published estimate from a study in Mumbai.</li>
          <li>The data show links, not proof that one thing causes another.</li>
          <li>Where the data suggested a bad habit adds years, the effect was set to zero.</li>
        </ul>
        {failure && <p role="alert" className="mt-4 font-semibold text-blood">{failure}</p>}
        <div className="mt-8 flex flex-wrap gap-3">
          <Button variant="blood" disabled={busy} onClick={download}>{busy ? 'Preparing…' : 'Download PDF'}</Button>
          <Button variant="line" onClick={onEdit}>Edit answers</Button>
          <Button variant="line" onClick={onClear}>Clear and close</Button>
        </div>
      </Page>
    </motion.main>
  )
}
