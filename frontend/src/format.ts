export function formatEur(value: number | null, placeholder = '—'): string {
  if (value === null) return placeholder
  return `€${value.toFixed(2)}`
}

export function formatPercent(value: number | null, placeholder = '—'): string {
  if (value === null) return placeholder
  const sign = value >= 0 ? '+' : ''
  return `${sign}${(value * 100).toFixed(1)}%`
}
