import { useState, useEffect, useRef } from 'react'
import type { CardEntry } from '../types'
import { formatEur } from '../format'

interface Props {
  cards: CardEntry[]
  onPredict: (uuid: string) => void
  isLoading: boolean
}

function cardMeta(card: CardEntry): string {
  return `${card.set_code} · ${card.rarity} · ${formatEur(card.eur, 'no price')}`
}

function cardLabel(card: CardEntry): string {
  return `${card.name} [${cardMeta(card)}]`
}

export function CardSearch({ cards, onPredict, isLoading }: Props) {
  const [query, setQuery] = useState('')
  const [selected, setSelected] = useState<CardEntry | null>(null)
  const [open, setOpen] = useState(false)
  const [focusedIndex, setFocusedIndex] = useState(-1)
  const containerRef = useRef<HTMLDivElement>(null)
  const listRef = useRef<HTMLUListElement>(null)

  const matches =
    query.length >= 2
      ? cards
          .filter(
            (c) =>
              c.name.toLowerCase().includes(query.toLowerCase()) ||
              c.set_code.toLowerCase().includes(query.toLowerCase()),
          )
          .slice(0, 10)
      : []

  useEffect(() => {
    function handleOutsideClick(e: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setOpen(false)
      }
    }
    document.addEventListener('mousedown', handleOutsideClick)
    return () => document.removeEventListener('mousedown', handleOutsideClick)
  }, [])

  // Scroll focused item into view
  useEffect(() => {
    if (focusedIndex >= 0 && listRef.current) {
      const item = listRef.current.children[focusedIndex] as HTMLElement | undefined
      item?.scrollIntoView({ block: 'nearest' })
    }
  }, [focusedIndex])

  function handleInputChange(e: React.ChangeEvent<HTMLInputElement>) {
    setQuery(e.target.value)
    setSelected(null)
    setFocusedIndex(-1)
    setOpen(true)
  }

  function handleSelect(card: CardEntry) {
    setQuery(cardLabel(card))
    setSelected(card)
    setFocusedIndex(-1)
    setOpen(false)
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (!open || matches.length === 0) return
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setFocusedIndex((prev) => Math.min(prev + 1, matches.length - 1))
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setFocusedIndex((prev) => Math.max(prev - 1, 0))
    } else if (e.key === 'Enter' && focusedIndex >= 0) {
      e.preventDefault()
      handleSelect(matches[focusedIndex])
    } else if (e.key === 'Escape') {
      setOpen(false)
    }
  }

  return (
    <div ref={containerRef} className="relative w-full max-w-md">
      <input
        type="text"
        value={query}
        onChange={handleInputChange}
        onKeyDown={handleKeyDown}
        placeholder="Type card name or set code (min. 2 characters)..."
        className="w-full rounded-lg border border-gray-300 px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
      />
      {open && matches.length > 0 && (
        <ul
          ref={listRef}
          className="absolute z-10 mt-1 max-h-64 w-full overflow-y-auto rounded-lg border border-gray-200 bg-white shadow-lg"
        >
          {matches.map((card, i) => (
            <li
              key={card.uuid}
              onMouseDown={() => handleSelect(card)}
              onMouseEnter={() => setFocusedIndex(i)}
              className={`cursor-pointer px-4 py-2 text-sm ${
                i === focusedIndex ? 'bg-indigo-100 text-indigo-900' : 'hover:bg-indigo-50'
              }`}
            >
              <span className="font-medium">{card.name}</span>
              <span className="ml-2 text-xs text-gray-400">{cardMeta(card)}</span>
            </li>
          ))}
        </ul>
      )}
      <button
        onClick={() => { if (selected) onPredict(selected.uuid) }}
        disabled={selected === null || isLoading}
        className="mt-3 w-full rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {isLoading ? 'Loading...' : 'Predict'}
      </button>
    </div>
  )
}
