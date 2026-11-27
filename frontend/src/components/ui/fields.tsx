import { useId, type ComponentProps } from 'react'
import { cn } from '../../lib/cn'

function Problem({ id, message }: { id: string; message?: string }) {
  return message ? <p id={id} role="alert" className="mt-1 text-sm font-semibold text-blood">{message}</p> : null
}

type FieldProps = Omit<ComponentProps<'input'>, 'onChange'> & {
  label: string
  hint?: string
  error?: string
  onChange: (value: string) => void
}

export function Field({ label, hint, error, onChange, className, ...props }: FieldProps) {
  const id = useId()
  return (
    <div className={className}>
      <label htmlFor={id} className="block font-semibold">{label}</label>
      {hint && <p id={`${id}-hint`} className="text-sm text-faded">{hint}</p>}
      <input
        id={id}
        aria-invalid={Boolean(error)}
        aria-describedby={cn(hint && `${id}-hint`, error && `${id}-error`) || undefined}
        className="mt-1 w-full border-0 border-b-2 border-pen/30 bg-transparent py-1 font-hand text-3xl focus:border-blood focus:outline-none"
        onChange={(event) => onChange(event.target.value)}
        {...props}
      />
      <Problem id={`${id}-error`} message={error} />
    </div>
  )
}

interface ChoiceProps {
  legend: string
  options: { value: string; label: string; hint?: string }[]
  value: string
  error?: string
  onChange: (value: string) => void
}

export function Choice({ legend, options, value, error, onChange }: ChoiceProps) {
  const name = useId()
  return (
    <fieldset aria-describedby={error ? `${name}-error` : undefined}>
      <legend className="font-semibold">{legend}</legend>
      <div className="mt-2 flex flex-wrap gap-2">
        {options.map((option) => (
          <label
            key={option.value}
            className="cursor-pointer rounded-sm border border-pen/30 bg-paper px-4 py-2 transition-colors has-checked:border-pen has-checked:bg-pen has-checked:text-paper has-focus-visible:outline-2 has-focus-visible:outline-blood"
          >
            <input
              type="radio" className="sr-only" name={name} value={option.value}
              checked={value === option.value} onChange={() => onChange(option.value)}
            />
            <span className="font-display text-lg font-bold">{option.label}</span>
            {option.hint && <span className="block text-xs opacity-70">{option.hint}</span>}
          </label>
        ))}
      </div>
      <Problem id={`${name}-error`} message={error} />
    </fieldset>
  )
}

export function Check({ label, checked, onChange }: { label: string; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <label className="flex min-h-11 cursor-pointer items-center gap-3">
      <input type="checkbox" className="size-5 accent-blood" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      <span>{label}</span>
    </label>
  )
}
