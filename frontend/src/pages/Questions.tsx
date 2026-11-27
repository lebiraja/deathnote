import { AnimatePresence, motion } from 'motion/react'
import { useState, type FormEvent } from 'react'
import { Button } from '../components/ui/button'
import { Check, Choice, Field } from '../components/ui/fields'
import { sleepHours, stepErrors, steps, type Draft, type Errors } from '../lib/form'
import { alcoholOptions, dietOptions, smokingOptions } from '../lib/labels'

interface Props {
  draft: Draft
  onChange: (draft: Draft) => void
  onExit: () => void
  onSubmit: () => Promise<void>
}

export function Questions({ draft, onChange, onExit, onSubmit }: Props) {
  const [step, setStep] = useState(0)
  const [errors, setErrors] = useState<Errors>({})
  const [failure, setFailure] = useState('')
  const [busy, setBusy] = useState(false)
  const last = step === steps.length - 1
  const hours = sleepHours(draft.bedtime, draft.wake_time)

  const set = <K extends keyof Draft>(key: K) => (value: Draft[K]) => {
    onChange({ ...draft, [key]: value })
    setErrors((current) => ({ ...current, [key]: undefined }))
  }

  const move = (next: number) => {
    setErrors({})
    setFailure('')
    setStep(next)
  }

  const submit = async (event: FormEvent) => {
    event.preventDefault()
    const found = stepErrors(draft, step)
    setErrors(found)
    if (Object.keys(found).length > 0) return
    if (!last) return move(step + 1)
    setBusy(true)
    setFailure('')
    try {
      await onSubmit()
    } catch (error) {
      setFailure(error instanceof Error ? error.message : 'Something went wrong. Try again.')
    } finally {
      setBusy(false)
    }
  }

  return (
    <main className="mx-auto max-w-2xl px-4 py-8">
      <p className="mb-4 flex justify-between text-sm text-paper/70">
        <span className="font-display text-2xl font-bold text-paper">deathnote</span>
        <span>Page {step + 1} of {steps.length}</span>
      </p>
      <form noValidate onSubmit={submit} className="sheet overflow-hidden py-8 pl-14 pr-5 sm:pr-10">
        <AnimatePresence mode="wait">
          <motion.div
            key={step}
            initial={{ opacity: 0, x: 24 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -24 }}
            transition={{ duration: 0.2 }}
            className="space-y-7"
          >
            <h1 className="font-display text-5xl font-bold">{steps[step]?.title}</h1>

            {step === 0 && (
              <>
                <Field label="Age" type="number" inputMode="numeric" value={draft.age} error={errors.age} onChange={set('age')} />
                <Choice legend="Sex" value={draft.sex} error={errors.sex} onChange={set('sex')}
                  options={[{ value: 'female', label: 'Female' }, { value: 'male', label: 'Male' }]} />
                <div className="grid gap-7 sm:grid-cols-2">
                  <Field label="Height (cm)" hint="5 ft 6 in is about 168 cm" type="number" inputMode="decimal"
                    value={draft.height_cm} error={errors.height_cm} onChange={set('height_cm')} />
                  <Field label="Weight (kg)" hint="Your usual weight" type="number" inputMode="decimal"
                    value={draft.weight_kg} error={errors.weight_kg} onChange={set('weight_kg')} />
                </div>
              </>
            )}

            {step === 1 && (
              <>
                <p>Think of a usual working day.</p>
                <div className="grid gap-7 sm:grid-cols-2">
                  <Field label="You fall asleep at" type="time" value={draft.bedtime} error={errors.bedtime} onChange={set('bedtime')} />
                  <Field label="You wake up at" type="time" value={draft.wake_time} error={errors.wake_time} onChange={set('wake_time')} />
                </div>
                <p aria-live="polite" className="font-hand text-5xl text-blood">
                  {hours === null ? ' ' : `${Number(hours.toFixed(2))} hours of sleep`}
                </p>
                <p className="text-sm text-faded">
                  Only the number of hours changes your estimate. The data cannot tell us what an early or late bedtime does.
                </p>
              </>
            )}

            {step === 2 && (
              <>
                <Choice legend="Smoking" options={smokingOptions} value={draft.smoking} error={errors.smoking} onChange={set('smoking')} />
                <Choice legend="Do you chew tobacco?" error={errors.chews_tobacco}
                  options={[{ value: 'no', label: 'No' }, { value: 'yes', label: 'Yes', hint: 'Gutka, khaini, paan with tobacco' }]}
                  value={draft.chews_tobacco === null ? '' : draft.chews_tobacco ? 'yes' : 'no'}
                  onChange={(v) => set('chews_tobacco')(v === 'yes')} />
                <Choice legend="Alcohol (one drink: a small peg, a glass of wine or a small beer)" options={alcoholOptions}
                  value={draft.alcohol} error={errors.alcohol} onChange={set('alcohol')} />
                <div className="grid gap-7 sm:grid-cols-2">
                  <Field label="Moderate exercise" hint="Minutes a week: brisk walking, cycling" type="number" inputMode="numeric"
                    value={draft.moderate_minutes} error={errors.moderate_minutes} onChange={set('moderate_minutes')} />
                  <Field label="Hard exercise" hint="Minutes a week: running, sport, gym" type="number" inputMode="numeric"
                    value={draft.vigorous_minutes} error={errors.vigorous_minutes} onChange={set('vigorous_minutes')} />
                </div>
                <Choice legend="How healthy is your diet overall?" options={dietOptions} value={draft.diet} error={errors.diet} onChange={set('diet')} />
              </>
            )}

            {step === 3 && (
              <>
                <fieldset>
                  <legend className="font-semibold">Has a doctor told you that you have any of these?</legend>
                  <Check label="Diabetes" checked={draft.diabetes} onChange={set('diabetes')} />
                  <Check label="High blood pressure" checked={draft.hypertension} onChange={set('hypertension')} />
                  <Check label="Heart disease or a heart attack" checked={draft.heart_disease} onChange={set('heart_disease')} />
                  <Check label="Asthma" checked={draft.asthma} onChange={set('asthma')} />
                </fieldset>
                <div className="grid gap-7 sm:grid-cols-2">
                  <Field label="Upper blood pressure" hint="Optional, e.g. 120" type="number" inputMode="numeric"
                    value={draft.systolic_bp} error={errors.systolic_bp} onChange={set('systolic_bp')} />
                  <Field label="Total cholesterol (mg/dL)" hint="Optional, from a blood test" type="number" inputMode="numeric"
                    value={draft.total_cholesterol} error={errors.total_cholesterol} onChange={set('total_cholesterol')} />
                </div>
              </>
            )}
          </motion.div>
        </AnimatePresence>

        {failure && <p role="alert" className="mt-6 font-semibold text-blood">{failure}</p>}
        <div className="mt-10 flex justify-between gap-3">
          <Button type="button" variant="line" onClick={() => (step === 0 ? onExit() : move(step - 1))}>Back</Button>
          <Button type="submit" disabled={busy} variant={last ? 'blood' : 'ink'}>
            {last ? (busy ? 'Writing…' : 'Read my page') : 'Next page'}
          </Button>
        </div>
      </form>
    </main>
  )
}
