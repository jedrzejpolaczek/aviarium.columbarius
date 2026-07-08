import { describe, it, expect } from 'vitest'
import { formatEur } from './format'

describe('formatEur', () => {
  it('formats a number with 2 decimals and a euro sign', () => {
    expect(formatEur(2.5)).toBe('€2.50')
  })

  it('returns the default placeholder for null', () => {
    expect(formatEur(null)).toBe('—')
  })

  it('returns a custom placeholder when given one', () => {
    expect(formatEur(null, 'no price')).toBe('no price')
  })

  it('returns an empty custom placeholder when given one', () => {
    expect(formatEur(null, '')).toBe('')
  })
})
