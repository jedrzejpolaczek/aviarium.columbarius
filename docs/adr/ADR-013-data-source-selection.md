# ADR-013: Data Source Selection

## Context

The pipeline needs several categories of data to support card price prediction:
- Card identity and metadata (oracle IDs, names, set info, legalities)
- Price history (multiple vendors and finishes)
- Format demand signals (how often a card appears in competitive decks)
- Tournament play data (actual top-8 placements)

Several external sources were evaluated for each category.

## Decision

| Category | Source | Access method |
|---|---|---|
| Card metadata + prices | Scryfall | JSON bulk download |
| Card UUIDs + extended prices | MTGJson | JSON bulk download |
| Format staples | MTGGoldfish | HTML scrape |
| Tournament results | MTGTop8 | HTML scrape (3-level) |

**Scryfall** — primary card metadata source. Provides oracle IDs, all printings, legalities, EDHREC rank, and prices (USD, EUR, foil variants) in a single bulk file (~300k records). Free, no auth required, updated daily.

**MTGJson** — secondary card source and price supplement. Provides stable UUIDs that link printings cross-vendor, plus CardMarket and TCGPlayer prices not available from Scryfall. Free bulk download.

**MTGGoldfish format staples** — deck-percentage signal per format. Widely used as a format meta barometer; `deck_pct` measures how many decks in a format run a card, which is a leading indicator of demand.

**MTGTop8 tournament results** — ground truth for competitive play. Provides structured top-8 decklists per tournament per format. Used to derive `top8_appearances` and `main_deck_pct` signals in the Gold layer.

## Consequences

### Positive
- All four sources are free with no API key or rate-limit agreements required.
- Scryfall + MTGJson bulk downloads avoid per-card request limits.
- MTGGoldfish and MTGTop8 cover demand signals unavailable from card databases.

### Negative
- MTGGoldfish and MTGTop8 are HTML scrapers — layout changes will break extraction silently.
- MTGTop8 requires 3-level scraping (format list → event → decklist), making it brittle.
- No official API contract for any HTML source; ToS must be monitored.

### Neutral
- CardMarket and TCGPlayer prices come indirectly via MTGJson, not their own APIs.
- EDHREC rank is available via Scryfall's `edhrec_rank` field — no separate source needed for Commander demand.

## Alternatives Considered

| Source | Reason not chosen |
|---|---|
| TCGPlayer API | Requires paid partnership / API key |
| CardMarket API | Requires authentication and is rate-limited; data covered by MTGJson |
| EDHREC scraper | `edhrec_rank` from Scryfall is sufficient for the Commander demand signal |
| MTGDecks | Overlaps with MTGTop8; MTGTop8 has more structured HTML and broader format coverage |
