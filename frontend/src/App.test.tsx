import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import { App } from './App'

describe('App', () => {
  it('opens the notebook and holds the first page until it is answered', async () => {
    const user = userEvent.setup()
    render(<App />)
    await user.click(screen.getByRole('button', { name: 'Open the notebook' }))
    await user.click(screen.getByRole('button', { name: 'Next page' }))
    expect(screen.getAllByRole('alert')).toHaveLength(4)
    expect(screen.getByRole('heading', { name: 'Who you are' })).toBeInTheDocument()
  })
})
