import { Slot } from '@radix-ui/react-slot'
import { cva, type VariantProps } from 'class-variance-authority'
import type { ComponentProps } from 'react'
import { cn } from '../../lib/cn'

const styles = cva(
  'inline-flex min-h-12 items-center justify-center rounded-sm px-6 font-display text-xl font-bold transition-colors disabled:cursor-wait disabled:opacity-50',
  {
    variants: {
      variant: {
        ink: 'bg-pen text-paper hover:bg-blood',
        blood: 'bg-blood text-paper hover:bg-pen',
        line: 'border border-pen/40 text-pen hover:border-blood hover:text-blood',
      },
    },
    defaultVariants: { variant: 'ink' },
  },
)

type Props = ComponentProps<'button'> & VariantProps<typeof styles> & { asChild?: boolean }

export function Button({ className, variant, asChild, ...props }: Props) {
  const Component = asChild ? Slot : 'button'
  return <Component className={cn(styles({ variant }), className)} {...props} />
}
