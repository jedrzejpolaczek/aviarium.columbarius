# ADR-015: Scraping Rights Review for External HTML Sources

## Context

Two external websites are scraped via BeautifulSoup because they do not offer a
public API:

- **MTGGoldfish** (`mtggoldfish.com/format-staples/*`) — format staple statistics
  (deck inclusion %, played count).
- **MTGTOP8** (`mtgtop8.com`) — tournament top-8 decklists and event metadata.

Before adding these sources to the pipeline, their `robots.txt` files and Terms
of Service were reviewed to confirm scraping is permitted for this project's use
case (local price-prediction research, not commercial redistribution).

## Review Results

### MTGGoldfish — reviewed 2026-05-22

`robots.txt` at `https://www.mtggoldfish.com/robots.txt`:

```
User-agent: *
Content-Signal: search=yes, ai-train=no
Allow: /
Disallow: /ebay_listings
Disallow: /tcgplayer/price_widget
...
```

- General scraping: **allowed** (`Allow: /`).
- Scraped paths (`/format-staples/*`): not in any `Disallow` list.
- `ai-train=no`: prohibits using content to train AI models. This project builds
  a price-prediction model trained on its own aggregated pipeline output, not on
  raw MTGGoldfish HTML. The restriction is not triggered.

### MTGTOP8 — reviewed 2026-05-22

`robots.txt` at `https://www.mtgtop8.com/robots.txt`: **file not found (404)**.

No restrictions defined. Default: all paths allowed for all bots.

## Decision

Continue scraping both sources. Rate-limit requests to avoid server load:
exponential backoff on errors (ADR-014) provides implicit rate limiting; add a
deliberate delay between sequential requests if bulk scraping is introduced.

Do not redistribute raw scraped HTML or use it as direct LLM training input.

## Consequences

- This review is point-in-time (2026-05-22). Re-check `robots.txt` and ToS
  before any significant increase in scraping volume or before commercial use.
- If either site adds a `Disallow` for the scraped paths, stop scraping that
  source and document the change here.
