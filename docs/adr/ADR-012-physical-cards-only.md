# ADR-012: Physical Cards Only — No Digital Formats or Tix Pricing

## Context

Magic: The Gathering exists in two distinct formats: physical (paper) and digital
(MTGO — Magic: The Gathering Online, and Arena). Digital cards have their own economy;
MTGO uses an in-game currency called **tix** (tickets) as the unit of exchange.

Scryfall price data includes a `tix` field alongside `usd`, `usd_foil`, `eur`, and
`eur_foil`. Some cards exist exclusively or predominantly in a digital format (e.g.
MTGO-only reprints, digital-only sets).

## Decision

This project targets the **physical (paper) card market only**. Digital card versions
and their `tix` pricing are explicitly out of scope.

- The `tix` column is captured in `bronze_scryfall_prices_history` as a scalar FLOAT
  column (ADR-025) but **excluded at the silver transformation step** and never appears
  in silver or gold outputs. The exclusion decision lives in Silver SQL, not Bronze.
- No filtering, ranking, or valuation logic will be built around `tix`.
- Cards that exist only in digital formats are not a supported use case.

## Consequences

### Positive
- Scope is tightly bounded to one market, keeping valuation models simpler and more
  interpretable.
- No need to model the MTGO economy (tix supply, redemption mechanics, bot pricing),
  which behaves very differently from the paper market.

### Negative
- Users interested in MTGO card values cannot use this pipeline without modification.
- If a card has paper price data missing but valid `tix` data, the gap is not filled.

### Neutral
- `tix` is stored in the Bronze Scryfall table (ADR-025) and explicitly excluded from
  Silver SQL extractions (`scryfall_prices_base.sql`, `scryfall_language_prices_base.sql`).

## Alternatives Considered

| Approach | Reason rejected |
|---|---|
| Include tix as an optional price dimension | Adds MTGO-specific complexity without a clear user need |
| Filter out digital-only cards at ingest | Premature — digital cards may share identifiers with paper reprints; safer to exclude at the metric layer |
