import { habitsSchema, profileSchema, type Habits, type Profile } from './schema'

/** Raw form answers: numbers stay as typed text until validated. */
export interface Draft {
  age: string
  sex: string
  height_cm: string
  weight_kg: string
  bedtime: string
  wake_time: string
  smoking: string
  chews_tobacco: boolean | null
  alcohol: string
  moderate_minutes: string
  vigorous_minutes: string
  diet: string
  diabetes: boolean
  hypertension: boolean
  heart_disease: boolean
  asthma: boolean
  systolic_bp: string
  total_cholesterol: string
}

export const emptyDraft: Draft = {
  age: '', sex: '', height_cm: '', weight_kg: '', bedtime: '23:00', wake_time: '06:30',
  smoking: '', chews_tobacco: null, alcohol: '', moderate_minutes: '', vigorous_minutes: '0', diet: '',
  diabetes: false, hypertension: false, heart_disease: false, asthma: false,
  systolic_bp: '', total_cholesterol: '',
}

export const steps = [
  { title: 'Who you are', fields: ['age', 'sex', 'height_cm', 'weight_kg'] },
  { title: 'Your nights', fields: ['bedtime', 'wake_time'] },
  { title: 'Your habits', fields: ['smoking', 'chews_tobacco', 'alcohol', 'moderate_minutes', 'vigorous_minutes', 'diet'] },
  { title: 'Your health', fields: ['diabetes', 'hypertension', 'heart_disease', 'asthma', 'systolic_bp', 'total_cholesterol'] },
] as const satisfies readonly { title: string; fields: readonly (keyof Draft)[] }[]

export type Errors = Partial<Record<keyof Draft, string>>

export function sleepHours(bedtime: string, wake: string): number | null {
  const toMinutes = (time: string): number => {
    const [h, m] = time.split(':').map(Number)
    return h === undefined || m === undefined ? Number.NaN : h * 60 + m
  }
  const total = (toMinutes(wake) - toMinutes(bedtime) + 1440) % 1440
  return bedtime && wake && Number.isFinite(total) ? total / 60 : null
}

const required = (text: string): number | undefined => (text.trim() === '' ? undefined : Number(text))
const optional = (text: string): number | null => (text.trim() === '' ? null : Number(text))

function toCandidate(draft: Draft) {
  return {
    ...draft,
    age: required(draft.age),
    height_cm: required(draft.height_cm),
    weight_kg: required(draft.weight_kg),
    chews_tobacco: draft.chews_tobacco ?? undefined,
    moderate_minutes: required(draft.moderate_minutes),
    vigorous_minutes: required(draft.vigorous_minutes),
    systolic_bp: optional(draft.systolic_bp),
    total_cholesterol: optional(draft.total_cholesterol),
  }
}

/** Errors for the fields of one step. */
export function stepErrors(draft: Draft, step: number): Errors {
  const fields: readonly string[] = steps[step]?.fields ?? []
  const errors: Errors = {}
  const result = profileSchema.safeParse(toCandidate(draft))
  for (const issue of result.error?.issues ?? []) {
    const field = String(issue.path[0])
    if (fields.includes(field)) errors[field as keyof Draft] ??= issue.message
  }
  if (fields.includes('wake_time')) {
    const hours = sleepHours(draft.bedtime, draft.wake_time)
    if (hours === null || hours < 2 || hours > 16) errors.wake_time = 'Sleep must be between 2 and 16 hours'
  }
  return errors
}

export const toProfile = (draft: Draft): Profile => profileSchema.parse(toCandidate(draft))
export const toHabits = (profile: Profile): Habits => habitsSchema.parse(profile)
