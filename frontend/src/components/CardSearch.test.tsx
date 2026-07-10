import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { CardSearch } from './CardSearch'
import type { CardEntry } from '../types'

const cards: CardEntry[] = [
  { uuid: '1', name: 'Lightning Bolt', set_code: 'LEA', rarity: 'common', eur: 5.5 },
  { uuid: '2', name: 'Lightning Strike', set_code: 'M19', rarity: 'common', eur: null },
]

describe('CardSearch label consistency', () => {
  it('dropdown item and post-selection input show the same set/rarity/price text', () => {
    render(<CardSearch cards={cards} onPredict={vi.fn()} isLoading={false} />)

    fireEvent.change(screen.getByPlaceholderText(/type card name/i), {
      target: { value: 'Light' },
    })

    const item = screen.getByText('LEA · common · €5.50')
    fireEvent.mouseDown(item)

    expect(screen.getByDisplayValue('Lightning Bolt [LEA · common · €5.50]')).toBeInTheDocument()
  })

  it('shows the "no price" placeholder for a null-price card in both the dropdown and the input', () => {
    render(<CardSearch cards={cards} onPredict={vi.fn()} isLoading={false} />)

    fireEvent.change(screen.getByPlaceholderText(/type card name/i), {
      target: { value: 'Light' },
    })

    const item = screen.getByText('M19 · common · no price')
    fireEvent.mouseDown(item)

    expect(screen.getByDisplayValue('Lightning Strike [M19 · common · no price]')).toBeInTheDocument()
  })
})
