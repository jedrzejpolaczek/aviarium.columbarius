import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { App } from './App'
import * as api from './api'
import type { CardEntry, PredictionResponse } from './types'

const mockCard: CardEntry = {
  uuid: 'u1',
  name: 'Lightning Bolt',
  set_code: 'LEA',
  rarity: 'common',
  eur: 1.5,
}

const mockPrediction: PredictionResponse = {
  card_name: 'Lightning Bolt',
  current_price: 1.2,
  predicted_price: 2.5,
  log_return_7d: 0.05,
  tier: 1,
  model_run_id: 'run-1',
}

async function selectCardAndSubmit() {
  const input = screen.getByPlaceholderText(/type card name or set code/i)
  fireEvent.change(input, { target: { value: 'Lightning' } })

  const option = await screen.findByText('Lightning Bolt')
  fireEvent.mouseDown(option)

  const button = screen.getByRole('button', { name: /predict/i })
  fireEvent.click(button)
}

beforeEach(() => {
  vi.restoreAllMocks()
})

describe('App', () => {
  it('renders the heading and card search input once fetchCards resolves', async () => {
    vi.spyOn(api, 'fetchCards').mockResolvedValue([mockCard])

    render(<App />)

    expect(screen.getByText('MTG Price Predictor')).toBeInTheDocument()
    expect(screen.getByPlaceholderText(/type card name or set code/i)).toBeInTheDocument()

    await waitFor(() => expect(api.fetchCards).toHaveBeenCalled())
  })

  it('does not crash when fetchCards rejects — autocomplete is just unavailable', async () => {
    vi.spyOn(api, 'fetchCards').mockRejectedValue(new Error('network error'))

    render(<App />)

    await waitFor(() => expect(api.fetchCards).toHaveBeenCalled())
    expect(screen.getByText('MTG Price Predictor')).toBeInTheDocument()
  })

  it('runs the full predict flow: select a card, submit, and render the prediction result', async () => {
    vi.spyOn(api, 'fetchCards').mockResolvedValue([mockCard])
    vi.spyOn(api, 'fetchPrediction').mockResolvedValue(mockPrediction)

    render(<App />)
    await waitFor(() => expect(api.fetchCards).toHaveBeenCalled())

    await selectCardAndSubmit()

    await waitFor(() => {
      expect(screen.getByText('€2.50')).toBeInTheDocument()
    })
    expect(api.fetchPrediction).toHaveBeenCalledWith('u1')
  })

  it('shows an error message when fetchPrediction rejects', async () => {
    vi.spyOn(api, 'fetchCards').mockResolvedValue([mockCard])
    vi.spyOn(api, 'fetchPrediction').mockRejectedValue(new Error('Card not found'))

    render(<App />)
    await waitFor(() => expect(api.fetchCards).toHaveBeenCalled())

    await selectCardAndSubmit()

    await waitFor(() => {
      expect(screen.getByText('Card not found')).toBeInTheDocument()
    })
  })
})
