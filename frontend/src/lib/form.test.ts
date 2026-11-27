import { describe, expect, it } from 'vitest'
import { emptyDraft, sleepHours, stepErrors, toHabits, toProfile, type Draft } from './form'

const filled: Draft = {
  ...emptyDraft, age: '30', sex: 'male', height_cm: '172', weight_kg: '65', smoking: 'never',
  chews_tobacco: false, alcohol: 'never', moderate_minutes: '200', diet: 'good',
}

describe('sleepHours', () => {
  it('wraps past midnight', () => expect(sleepHours('23:00', '06:30')).toBe(7.5))
  it('handles daytime sleep', () => expect(sleepHours('09:00', '16:00')).toBe(7))
  it('is unknown without a time', () => expect(sleepHours('', '06:00')).toBeNull())
})

describe('validation', () => {
  it('reports only the current step', () => {
    expect(Object.keys(stepErrors(emptyDraft, 0)).sort()).toEqual(['age', 'height_cm', 'sex', 'weight_kg'])
  })
  it('accepts a complete draft on every step', () => {
    for (const step of [0, 1, 2, 3]) expect(stepErrors(filled, step)).toEqual({})
  })
  it('rejects ages outside the data', () => expect(stepErrors({ ...filled, age: '80' }, 0).age).toBeDefined())
  it('rejects implausible sleep', () => {
    expect(stepErrors({ ...filled, wake_time: '23:30' }, 1).wake_time).toBeDefined()
  })
  it('sends blank optional readings as null', () => {
    expect(toProfile(filled)).toMatchObject({ age: 30, systolic_bp: null, total_cholesterol: null })
  })
  it('keeps only changeable answers for what-if', () => {
    const habits = toHabits(toProfile(filled))
    expect(habits).not.toHaveProperty('age')
    expect(habits).toHaveProperty('bedtime', '23:00')
  })
})
