import { describe, it, expect } from 'vitest'
import { formatEur, formatPercent } from './format'

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

describe('formatPercent', () => {
  it('formats a positive value with a leading + sign', () => {
    expect(formatPercent(0.05)).toBe('+5.0%')
  })

  it('formats a negative value without a double sign', () => {
    expect(formatPercent(-0.123)).toBe('-12.3%')
  })

  it('formats zero with a leading + sign', () => {
    expect(formatPercent(0)).toBe('+0.0%')
  })

  it('returns the default placeholder for null', () => {
    expect(formatPercent(null)).toBe('—')
  })
})
