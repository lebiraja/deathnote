import type { CurvePoint } from '../lib/schema'

const W = 600
const H = 260
const PAD = { left: 40, right: 12, top: 10, bottom: 28 }

interface Props {
  curve: CurvePoint[]
  labels: { you: string; average: string }
}

export function SurvivalChart({ curve, labels }: Props) {
  const first = curve[0]?.age ?? 20
  const last = curve.at(-1)?.age ?? 100
  const x = (age: number) => PAD.left + ((age - first) / (last - first)) * (W - PAD.left - PAD.right)
  const y = (share: number) => PAD.top + (1 - share) * (H - PAD.top - PAD.bottom)
  const path = (key: 'you' | 'average') => curve.map((p, i) => `${i ? 'L' : 'M'}${x(p.age)} ${y(p[key])}`).join('')
  const decades = curve.map((p) => p.age).filter((age) => age % 10 === 0)
  const atAge = (age: number) => curve.find((p) => p.age === age)

  return (
    <figure>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="Chance of being alive at each age">
        {[0, 0.25, 0.5, 0.75, 1].map((share) => (
          <g key={share}>
            <line x1={PAD.left} x2={W - PAD.right} y1={y(share)} y2={y(share)} className="stroke-pen/15" />
            <text x={PAD.left - 6} y={y(share) + 4} textAnchor="end" className="fill-faded text-[11px]">{share * 100}%</text>
          </g>
        ))}
        {decades.map((age) => (
          <text key={age} x={x(age)} y={H - 8} textAnchor="middle" className="fill-faded text-[11px]">{age}</text>
        ))}
        <path d={path('average')} fill="none" strokeDasharray="5 4" strokeWidth="1.5" className="stroke-pen" />
        <path d={path('you')} fill="none" strokeWidth="3" className="stroke-blood" />
      </svg>
      <figcaption className="flex flex-wrap gap-5 text-sm">
        <span><span className="mr-2 inline-block h-1 w-6 bg-blood align-middle" />{labels.you}</span>
        <span><span className="mr-2 inline-block w-6 border-t-2 border-dashed border-pen align-middle" />{labels.average}</span>
      </figcaption>
      <table className="sr-only">
        <caption>Chance of being alive by age</caption>
        <thead><tr><th>Age</th><th>{labels.you}</th><th>{labels.average}</th></tr></thead>
        <tbody>
          {decades.map((age) => (
            <tr key={age}>
              <td>{age}</td>
              <td>{Math.round((atAge(age)?.you ?? 0) * 100)}%</td>
              <td>{Math.round((atAge(age)?.average ?? 0) * 100)}%</td>
            </tr>
          ))}
        </tbody>
      </table>
    </figure>
  )
}
