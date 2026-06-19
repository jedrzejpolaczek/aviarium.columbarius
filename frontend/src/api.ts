import type { CardEntry, PredictionResponse } from './types'

const BASE_URL = (import.meta.env.VITE_API_URL as string | undefined) ?? 'http://localhost:8000'

export async function fetchCards(): Promise<CardEntry[]> {
  const res = await fetch(`${BASE_URL}/cards`)
  if (!res.ok) throw new Error(`Failed to fetch cards: ${res.status}`)
  const data = await res.json() as { cards: CardEntry[] }
  return data.cards
}

export async function fetchPrediction(uuid: string): Promise<PredictionResponse> {
  const res = await fetch(`${BASE_URL}/predict/uuid/${encodeURIComponent(uuid)}`)
  if (res.status === 404) throw new Error('Card not found')
  if (res.status === 503) throw new Error('Model not loaded — set MODEL_RUN_ID')
  if (!res.ok) throw new Error(`Prediction failed: ${res.status}`)
  return res.json() as Promise<PredictionResponse>
}
