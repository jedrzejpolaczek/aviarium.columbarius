import { useState, useEffect } from 'react'
import { CardSearch } from './components/CardSearch'
import { PredictionResult } from './components/PredictionResult'
import { fetchCards, fetchPrediction } from './api'
import type { CardEntry, PredictionResponse } from './types'

type AppState =
  | { status: 'idle' }
  | { status: 'loading' }
  | { status: 'success'; data: PredictionResponse }
  | { status: 'error'; message: string }

export function App() {
  const [cards, setCards] = useState<CardEntry[]>([])
  const [state, setState] = useState<AppState>({ status: 'idle' })

  useEffect(() => {
    fetchCards()
      .then(setCards)
      .catch(() => {
        // Cards load silently — autocomplete will be unavailable
      })
  }, [])

  async function handlePredict(cardName: string) {
    setState({ status: 'loading' })
    try {
      const data = await fetchPrediction(cardName)
      setState({ status: 'success', data })
    } catch (err) {
      setState({
        status: 'error',
        message: err instanceof Error ? err.message : 'Unknown error',
      })
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 px-4 py-16">
      <div className="mx-auto max-w-lg">
        <h1 className="mb-2 text-3xl font-bold text-gray-900">MTG Price Predictor</h1>
        <p className="mb-8 text-sm text-gray-500">
          7-day price prediction for Magic: The Gathering cards
        </p>

        <CardSearch
          cards={cards}
          onPredict={handlePredict}
          isLoading={state.status === 'loading'}
        />

        {state.status === 'success' && <PredictionResult result={state.data} />}

        {state.status === 'error' && (
          <p className="mt-6 rounded-lg bg-red-50 px-4 py-3 text-sm text-red-600">
            {state.message}
          </p>
        )}
      </div>
    </div>
  )
}
