import { useEffect, useState } from 'react'
import { whatIf } from '../lib/api'
import { toHabits } from '../lib/form'
import { alcoholOptions, dietOptions, smokingOptions } from '../lib/labels'
import { habitsSchema, type Habits, type Profile, type WhatIf as Result } from '../lib/schema'
import { SurvivalChart } from './SurvivalChart'
import { Button } from './ui/button'
import { Choice, Field } from './ui/fields'
import { Slider } from './ui/slider'

const drinking = ['light', 'moderate', 'heavy']

/** Try different habits and see how the estimate moves. */
export function WhatIf({ profile }: { profile: Profile }) {
  const [changes, setChanges] = useState<Habits>(() => toHabits(profile))
  const [result, setResult] = useState<Result | null>(null)
  const [failure, setFailure] = useState('')
  const set = <K extends keyof Habits>(key: K) => (value: Habits[K]) => setChanges((c) => ({ ...c, [key]: value }))

  useEffect(() => {
    const controller = new AbortController()
    const timer = setTimeout(() => {
      whatIf(profile, changes, controller.signal)
        .then((next) => { setResult(next); setFailure('') })
        .catch((error: unknown) => {
          if (controller.signal.aborted) return
          setResult(null)
          setFailure(error instanceof Error ? error.message : 'Could not recalculate.')
        })
    }, 250)
    return () => { clearTimeout(timer); controller.abort() }
  }, [profile, changes])

  const change = result?.change_years ?? 0
  return (
    <div className="space-y-6">
      <div aria-live="polite" className="min-h-16">
        {failure ? <p className="font-semibold text-blood">{failure}</p> : result && (
          <p className="font-hand text-4xl">
            {change === 0 ? 'No change yet' : (
              <span className={change > 0 ? 'text-moss' : 'text-blood'}>
                {change > 0 ? '+' : '−'}{Math.abs(change)} years, middle estimate {result.after.median_age}
              </span>
            )}
          </p>
        )}
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        <Field label="Fall asleep at" type="time" value={changes.bedtime}
          onChange={(v) => habitsSchema.shape.bedtime.safeParse(v).success && set('bedtime')(v)} />
        <Field label="Wake up at" type="time" value={changes.wake_time}
          onChange={(v) => habitsSchema.shape.wake_time.safeParse(v).success && set('wake_time')(v)} />
      </div>
      <Slider label="Moderate exercise a week" unit="min" min={0} max={Math.max(600, profile.moderate_minutes)} step={10}
        value={changes.moderate_minutes} onChange={set('moderate_minutes')} />
      <Slider label="Hard exercise a week" unit="min" min={0} max={Math.max(300, profile.vigorous_minutes)} step={5}
        value={changes.vigorous_minutes} onChange={set('vigorous_minutes')} />
      <Slider label="Weight" unit="kg" min={30} max={Math.max(150, profile.weight_kg)} step={1}
        value={changes.weight_kg} onChange={set('weight_kg')} />
      <Choice legend="Diet" options={dietOptions} value={changes.diet}
        onChange={(v) => set('diet')(habitsSchema.shape.diet.parse(v))} />
      {profile.smoking !== 'never' && (
        <Choice legend="Smoking" options={smokingOptions.filter((o) => o.value !== 'never')} value={changes.smoking}
          onChange={(v) => set('smoking')(habitsSchema.shape.smoking.parse(v))} />
      )}
      {profile.chews_tobacco && (
        <Choice legend="Chewing tobacco" options={[{ value: 'yes', label: 'Keep chewing' }, { value: 'no', label: 'Stop' }]}
          value={changes.chews_tobacco ? 'yes' : 'no'} onChange={(v) => set('chews_tobacco')(v === 'yes')} />
      )}
      {drinking.includes(profile.alcohol) && (
        <Choice legend="Alcohol" options={alcoholOptions.filter((o) => drinking.includes(o.value))} value={changes.alcohol}
          onChange={(v) => set('alcohol')(habitsSchema.shape.alcohol.parse(v))} />
      )}

      {result && <SurvivalChart curve={result.curve} labels={{ you: 'With these changes', average: 'As you are now' }} />}
      <Button variant="line" onClick={() => setChanges(toHabits(profile))}>Reset changes</Button>
      <p className="text-sm text-faded">
        These are differences between groups of people. Changing a habit may not give you the same number of years.
      </p>
    </div>
  )
}
