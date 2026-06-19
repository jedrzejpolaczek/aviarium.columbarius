# ML Models — Findings

Wyniki i obserwacje z notebookow ML (Miesiac 2).
Wypelnij kazda sekcje po uruchomieniu odpowiedniego notebooka.

---

## T5 — Feature Engineering

**Notebook:** `01_feature_engineering.ipynb`
**Status:** [ ] Do uruchomienia

Kluczowe pytania do odpowiedzi:
- Ile kart ma kompletne lag features (nie-NaN dla lag_7d)?
- Jaki jest rozklad momentum_7d — czy symetryczny wokol 0?
- Czy rolling_std_14d dobrze rozroznia stabilne karty od spekulatywnych?

_(wypelnij po uruchomieniu notebooka)_

---

## T6 — Baseline vs LightGBM

**Notebook:** `02_baseline_lightgbm.ipynb`
**Status:** [ ] Data-gated (wymaga >= 8 snapshotow ~2026-06-17)

Kluczowe pytania:
- Naive MAE per tier = oficjalny prog do pobicia
- MA7d vs Naive: mean reversion czy momentum?
- AR1 vs Naive: czy autocorrelacja z Ljung-Box jest przewidywalna?
- LightGBM MAPE per tier vs Naive MAPE per tier

_(wypelnij po uruchomieniu notebooka)_

---

## T7 — Time Series (data-gated)

**Notebook:** `03_time_series.ipynb`
**Status:** [ ] Data-gated (wymaga >= 20 snapshotow ~2026-07-03)

UWAGA: Zamien kolejnosc z T8 — zrob T8 najpierw na dostepnych danych,
wróc do T7 gdy beda >= 20 snapshoty.

Kluczowe pytania:
- Prophet vs LightGBM: ktory model jest lepszy dla plynnych kart?
- Czy jest widoczna sezonowosc tygodniowa (FNM)?

_(wypelnij po uruchomieniu notebooka)_

---

## T8 — Optuna + SHAP

**Notebook:** `04_shap_optuna.ipynb`
**Status:** [ ] Do uruchomienia (nie wymaga pelnego CV, mozna na crosssectional)

Kluczowe pytania:
- Najlepsze parametry z Optuna (num_leaves, learning_rate, min_child_samples)?
- Kolejnosc waznosci SHAP: czy edhrec_saltiness > is_reserved?
- SHAP print_count: czy spada do zera gdy saltiness jest w modelu? (weryfikacja BA-02)
- Waterfall dla 3 kart: taniej common, Reserved List, tournament staple

_(wypelnij po uruchomieniu notebooka)_
