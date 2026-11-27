import { motion } from 'motion/react'
import type { Estimate } from '../lib/schema'

interface Props {
  age: number
  estimate: Estimate
  average: Estimate
}

/** A life drawn as one ink line, with the middle half of outcomes marked. */
export function Lifeline({ age, estimate, average }: Props) {
  const end = Math.max(100, Math.ceil(estimate.upper_age / 10) * 10, Math.ceil(average.median_age / 10) * 10)
  const at = (value: number) => `${(value / end) * 100}%`
  const label = `Lifeline from birth to ${end}. You are ${age}. Middle estimate ${estimate.median_age}, ` +
    `middle half ${estimate.lower_age} to ${estimate.upper_age}. Average ${average.median_age}.`
  return (
    <div role="img" aria-label={label} className="relative h-28">
      <div className="absolute top-12 h-px w-full bg-pen/30" />
      <motion.div className="absolute top-[46px] h-1 bg-pen" initial={{ width: 0 }} animate={{ width: at(age) }} transition={{ duration: 0.9 }} />
      <motion.div
        className="absolute top-9 h-6 bg-blood/25"
        style={{ left: at(estimate.lower_age) }}
        initial={{ width: 0 }} animate={{ width: at(estimate.upper_age - estimate.lower_age) }}
        transition={{ delay: 0.9, duration: 0.7 }}
      />
      <div className="absolute top-7 h-10 w-0.5 bg-blood" style={{ left: at(estimate.median_age) }} />
      <span className="absolute top-0 -translate-x-1/2 font-hand text-2xl text-blood" style={{ left: at(estimate.median_age) }}>
        {Math.round(estimate.median_age)}
      </span>
      <div className="absolute top-12 h-8 border-l border-dashed border-pen" style={{ left: at(average.median_age) }} />
      <span className="absolute top-20 -translate-x-1/2 whitespace-nowrap text-xs text-faded" style={{ left: at(average.median_age) }}>
        average {Math.round(average.median_age)}
      </span>
      <span className="absolute top-14 -translate-x-1/2 text-xs font-semibold" style={{ left: at(age) }}>now {age}</span>
    </div>
  )
}
