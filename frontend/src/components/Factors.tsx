import { motion } from 'motion/react'
import { factorLabels } from '../lib/labels'
import type { Prediction } from '../lib/schema'

export function Factors({ factors }: { factors: Prediction['factors'] }) {
  if (factors.length === 0) return <p>None of your answers differ much from the comparison person.</p>
  const largest = Math.max(...factors.map((f) => Math.abs(f.years)))
  return (
    <ul className="space-y-3">
      {factors.map((factor) => (
        <li key={factor.key}>
          <div className="flex justify-between gap-4">
            <span>{factorLabels[factor.key] ?? factor.key}</span>
            <span className={`font-hand text-2xl ${factor.years < 0 ? 'text-blood' : 'text-moss'}`}>
              {factor.years > 0 ? '+' : '−'}{Math.abs(factor.years)} yr
            </span>
          </div>
          <motion.div
            aria-hidden className={`h-1.5 ${factor.years < 0 ? 'bg-blood' : 'bg-moss'}`}
            initial={{ width: 0 }} animate={{ width: `${(Math.abs(factor.years) / largest) * 100}%` }}
          />
        </li>
      ))}
    </ul>
  )
}
