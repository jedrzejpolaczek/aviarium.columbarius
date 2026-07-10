import type { PredictionResponse } from '../types'
import { formatEur, formatPercent } from '../format'

interface Props {
  result: PredictionResponse
}

export function PredictionResult({ result }: Props) {
  const returnPositive = result.log_return_7d !== null && result.log_return_7d >= 0

  return (
    <div className="mt-8 w-full max-w-md rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
      <h2 className="text-lg font-semibold text-gray-800">{result.card_name}</h2>

      {result.predicted_price === null ? (
        <p className="mt-4 text-sm text-gray-500">
          Price too high for ML prediction — check Cardmarket directly.
        </p>
      ) : (
        <div className="mt-4 text-center">
          <p className="text-5xl font-bold text-indigo-600">
            {formatEur(result.predicted_price)}
          </p>
          <p className="mt-1 text-sm text-gray-400">predicted price (7d)</p>
        </div>
      )}

      <div className="mt-6 grid grid-cols-3 gap-4 border-t border-gray-100 pt-4">
        <div className="text-center">
          <p className="text-xs text-gray-400">Current price</p>
          <p className="mt-1 font-medium text-gray-700">{formatEur(result.current_price)}</p>
        </div>
        <div className="text-center">
          <p className="text-xs text-gray-400">Tier</p>
          <p className="mt-1 font-medium text-gray-700">{result.tier}</p>
        </div>
        <div className="text-center">
          <p className="text-xs text-gray-400">7d return</p>
          <p className={`mt-1 font-medium ${returnPositive ? 'text-green-600' : 'text-red-600'}`}>
            {formatPercent(result.log_return_7d)}
          </p>
        </div>
      </div>

      <p className="mt-4 text-xs text-gray-400">
        Model: {result.model_run_id !== '' ? result.model_run_id : '—'}
      </p>
    </div>
  )
}
