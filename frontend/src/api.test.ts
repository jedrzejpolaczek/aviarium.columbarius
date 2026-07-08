import { describe, it, expect, vi, beforeEach } from 'vitest'
import { fetchCards, fetchPrediction } from './api'

beforeEach(() => {
  vi.stubGlobal('fetch', vi.fn())
})

describe('fetchCards', () => {
  it('calls the /cards endpoint and returns the parsed card list', async () => {
    const mockCards = [
      { uuid: 'u1', name: 'Lightning Bolt', set_code: 'LEA', rarity: 'common', eur: 1.5 },
    ]
    ;(fetch as any).mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ cards: mockCards }),
    })

    const result = await fetchCards()

    expect(fetch).toHaveBeenCalledWith(expect.stringContaining('/cards'))
    expect(result).toEqual(mockCards)
  })

  it('throws when the response is not ok', async () => {
    ;(fetch as any).mockResolvedValue({ ok: false, status: 500, json: async () => ({}) })

    await expect(fetchCards()).rejects.toThrow('Failed to fetch cards: 500')
  })
})

describe('fetchPrediction', () => {
  it('calls /predict/uuid/<uuid> and returns the parsed prediction', async () => {
    const mockPrediction = {
      card_name: 'Lightning Bolt',
      current_price: 1.2,
      predicted_price: 2.5,
      log_return_7d: 0.05,
      tier: 1,
      model_run_id: 'run-1',
    }
    ;(fetch as any).mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => mockPrediction,
    })

    const result = await fetchPrediction('uuid-1')

    expect(fetch).toHaveBeenCalledWith(expect.stringContaining('uuid-1'))
    expect(result).toEqual(mockPrediction)
  })

  it('throws "Card not found" on a 404 response', async () => {
    ;(fetch as any).mockResolvedValue({ ok: false, status: 404, json: async () => ({}) })

    await expect(fetchPrediction('missing')).rejects.toThrow('Card not found')
  })

  it('throws a model-not-loaded message on a 503 response', async () => {
    ;(fetch as any).mockResolvedValue({ ok: false, status: 503, json: async () => ({}) })

    await expect(fetchPrediction('u1')).rejects.toThrow('Model not loaded')
  })

  it('throws a generic message for other non-ok statuses', async () => {
    ;(fetch as any).mockResolvedValue({ ok: false, status: 500, json: async () => ({}) })

    await expect(fetchPrediction('u1')).rejects.toThrow('Prediction failed: 500')
  })
})
