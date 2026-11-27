import * as Radix from '@radix-ui/react-slider'

interface Props {
  label: string
  value: number
  min: number
  max: number
  step: number
  unit: string
  onChange: (value: number) => void
}

export function Slider({ label, value, min, max, step, unit, onChange }: Props) {
  return (
    <div>
      <div className="flex items-baseline justify-between">
        <span className="font-semibold">{label}</span>
        <span className="font-hand text-2xl">{value} {unit}</span>
      </div>
      <Radix.Root
        className="relative flex h-11 touch-none items-center"
        value={[value]} min={min} max={max} step={step}
        onValueChange={([next]) => next !== undefined && onChange(next)}
      >
        <Radix.Track className="relative h-0.5 grow bg-pen/25">
          <Radix.Range className="absolute h-full bg-blood" />
        </Radix.Track>
        <Radix.Thumb aria-label={label} className="block size-6 rounded-full border-2 border-paper bg-pen shadow" />
      </Radix.Root>
    </div>
  )
}
