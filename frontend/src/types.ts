export interface CardEntry {
  uuid: string
  name: string
  set_code: string
  rarity: string
  eur: number | null
}

export interface PredictionResponse {
  card_name: string
  current_price: number | null
  predicted_price: number | null
  log_return_7d: number | null
  tier: number
  model_run_id: string
}
