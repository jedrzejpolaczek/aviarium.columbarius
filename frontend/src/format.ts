export function formatEur(value: number | null, placeholder = '—'): string {
  if (value === null) return placeholder
  return `€${value.toFixed(2)}`
}
