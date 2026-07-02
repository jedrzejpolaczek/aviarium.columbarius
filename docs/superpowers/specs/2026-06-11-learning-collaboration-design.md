# Plan współpracy i nauki — aviarium.columbarius

**Data:** 2026-06-11  
**Cel:** Zmiana sposobu współpracy tak, żeby użytkownik rozumiał projekt i budował intuicję programistyczną, zamiast tylko obserwować gotowy kod.

## Kontekst

Użytkownik:
- Doświadczony z Pythonem i sieciami neuronowymi (implementuje od zera)
- Nie zna drzew decyzyjnych i gradient boostingu teoretycznie
- Gubi się w bibliotekach: LightGBM, SHAP, Optuna, duckdb
- Projekt już istnieje (napisany z pomocą AI) — chce go zrozumieć i przepisać

## Podejście: Koncepcja → Analiza → Implementacja

Każda warstwa projektu przebiega w 4 krokach:

1. **Koncept** — tłumaczenie idei słowami, bez kodu, z analogiami
2. **Analiza istniejącego kodu** — wspólne czytanie, tłumaczenie każdej niejasnej linii
3. **Mini-spec + samodzielna implementacja** — użytkownik pisze od zera na podstawie specyfikacji
4. **Porównanie** — zestawienie jego kodu z istniejącym, omówienie wyborów

## Mapa nauki

| Etap | Temat | Kluczowe pliki |
|------|-------|----------------|
| 1 | Bronze — ingestion pipeline, DuckDB storage | `src/data/cards/storage/bronze/` |
| 2 | Silver — cleaning, normalization, joins | `src/data/cards/storage/silver/` |
| 3 | Gold — feature engineering, ML dataset | `src/data/cards/storage/gold/` |
| 4 | ML Features — lag features, sklearn pipeline | `src/ml/features/` |
| 5 | Modele: LightGBM, tiered, baseline | `src/ml/models/` |
| 6 | Trening i eksperymenty: trainer + MLflow | `src/ml/training/`, `scripts/` |
| 7 | Interpretacja modelu: SHAP, error analysis | `src/ml/evaluation/` |
| 8 | Rekomendacje: embeddingi i niedowartościowane karty | `src/ml/recommendation/` |
| 9 | Monitoring: drift, MAPE, automatyczny retrain | `src/monitoring/` |
| 10 | API: FastAPI — endpointy, schematy, zależności | `app/` |
| 11 | Frontend: interfejs użytkownika | `frontend/` |
| 12 | Produkcja: Docker, CI/CD, Makefile | `docker/`, `Makefile` |

## Zasady współpracy

**AI (Claude) zobowiązuje się:**
- Nie pisać kodu dopóki nie wyjaśnił konceptu
- W fazie implementacji dawać spec, nie rozwiązanie
- Na pytanie "jak to napisać?" — odpowiadać wskazówką
- Każdą nową bibliotekę tłumaczyć w jednym zdaniu przy pierwszym użyciu

**Użytkownik:**
- Pisze kod sam w fazie implementacji
- Pyta przed implementacją gdy czegoś nie rozumie
- Może w każdej chwili powiedzieć "nie rozumiem X"

**Wyjątki — kiedy AI pisze kod:**
- Boilerplate bez wartości edukacyjnej (np. konfiguracja połączenia z bazą)
- Fragmenty wymagające 2h+ nauki pobocznego tematu
- Po 15+ minutach frustracji — wtedy wskazówka, nie rozwiązanie

## Postęp

- [x] Etap 1: Bronze — ukończony 2026-06-15
- [x] Etap 2: Silver — ukończony 2026-06-16
- [x] Etap 3: Gold — ukończony 2026-06-20
- [x] Etap 4: ML Features — ukończony 2026-07-02
- [ ] Etap 5: Modele
- [ ] Etap 6: Trening
- [ ] Etap 7: SHAP
- [ ] Etap 8: Rekomendacje
- [ ] Etap 9: Monitoring
- [ ] Etap 10: API
- [ ] Etap 11: Frontend
- [ ] Etap 12: Produkcja

## Punkt startowy

Etap 1: Bronze — jak surowe dane z Cardmarket/Scryfall lądują w `data/bronze/cards.duckdb`.
