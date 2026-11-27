import { z } from 'zod'
import { predictionSchema, whatIfSchema, type Habits, type Prediction, type Profile, type WhatIf } from './schema'

const errorSchema = z.object({
  detail: z.union([z.string(), z.array(z.object({ msg: z.string() }))]),
})

async function post(path: string, body: unknown, signal?: AbortSignal): Promise<Response> {
  const response = await fetch(`/api/v1${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  })
  if (response.ok) return response
  const parsed = errorSchema.safeParse(await response.json().catch(() => null))
  if (!parsed.success) throw new Error(`The server could not answer (${response.status}). Try again.`)
  const { detail } = parsed.data
  throw new Error(typeof detail === 'string' ? detail : detail.map((d) => d.msg).join('. '))
}

export async function predict(profile: Profile): Promise<Prediction> {
  const response = await post('/predictions/predict', profile)
  return predictionSchema.parse(await response.json())
}

export async function whatIf(profile: Profile, changes: Habits, signal: AbortSignal): Promise<WhatIf> {
  const response = await post('/predictions/what-if', { profile, changes }, signal)
  return whatIfSchema.parse(await response.json())
}

export async function report(profile: Profile): Promise<Blob> {
  return (await post('/report', profile)).blob()
}
