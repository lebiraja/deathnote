import { z } from 'zod'

// The page's security policy forbids eval, which Zod would otherwise try.
z.config({ jitless: true })

const clock = z.string().regex(/^([01]\d|2[0-3]):[0-5]\d$/, 'Enter a time')
const number = (min: number, max: number) =>
  z.number({ error: 'Enter a number' }).min(min, `At least ${min}`).max(max, `At most ${max}`)
const minutes = number(0, 3000).int('Use whole minutes')

export const habitsSchema = z.object({
  weight_kg: number(30, 250),
  bedtime: clock,
  wake_time: clock,
  smoking: z.enum(['never', 'former', 'current'], { error: 'Choose one' }),
  chews_tobacco: z.boolean({ error: 'Choose one' }),
  alcohol: z.enum(['never', 'former', 'light', 'moderate', 'heavy'], { error: 'Choose one' }),
  moderate_minutes: minutes,
  vigorous_minutes: minutes,
  diet: z.enum(['excellent', 'very_good', 'good', 'fair', 'poor'], { error: 'Choose one' }),
})

export const profileSchema = habitsSchema.extend({
  age: number(20, 79).int('Use whole years'),
  sex: z.enum(['female', 'male'], { error: 'Choose one' }),
  height_cm: number(120, 220),
  diabetes: z.boolean(),
  hypertension: z.boolean(),
  heart_disease: z.boolean(),
  asthma: z.boolean(),
  systolic_bp: number(70, 260).nullable(),
  total_cholesterol: number(80, 500).nullable(),
})

const estimateSchema = z.object({
  median_age: z.number(),
  lower_age: z.number(),
  upper_age: z.number(),
  years_remaining: z.number(),
})
const curveSchema = z.array(z.object({ age: z.number(), you: z.number(), average: z.number() }))

export const predictionSchema = z.object({
  estimate: estimateSchema,
  average: estimateSchema,
  sleep_hours: z.number(),
  bmi: z.number(),
  curve: curveSchema,
  factors: z.array(z.object({ key: z.string(), years: z.number() })),
  advice: z.array(z.object({ title: z.string(), detail: z.string() })),
  model_version: z.string(),
})

export const whatIfSchema = z.object({
  before: estimateSchema,
  after: estimateSchema,
  change_years: z.number(),
  curve: curveSchema,
  sleep_hours: z.number(),
  bmi: z.number(),
})

export type Habits = z.infer<typeof habitsSchema>
export type Profile = z.infer<typeof profileSchema>
export type Estimate = z.infer<typeof estimateSchema>
export type Prediction = z.infer<typeof predictionSchema>
export type WhatIf = z.infer<typeof whatIfSchema>
export type CurvePoint = z.infer<typeof curveSchema>[number]
