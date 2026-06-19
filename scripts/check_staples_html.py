"""Check whether MTGGoldfish format-staples pages are JS-rendered or static.

Downloads one format-staples page with requests (same as the pipeline) and
counts the rows in table.table-staples. If the count is ~10, the full list
is JavaScript-rendered and requests.get only gets the skeleton. If it's much
higher, the pipeline extractor is producing the correct full list.

Usage:
    uv run scripts/check_staples_html.py
    uv run scripts/check_staples_html.py legacy     # test a specific format
"""

import sys
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

FORMAT = sys.argv[1] if len(sys.argv) > 1 else "modern"
URL = f"https://www.mtggoldfish.com/format-staples/{FORMAT}"
OUT = Path(f"data/raw/debug_staples_{FORMAT}.html")

print(f"Fetching {URL} ...")
response = httpx.get(
    URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=15, follow_redirects=True
)
response.raise_for_status()

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(response.text, encoding="utf-8")
print(f"Saved {len(response.text):,} bytes → {OUT}")

soup = BeautifulSoup(response.text, "lxml")
table = soup.select_one("table.table-staples")

if table is None:
    print("ERROR: table.table-staples not found — page structure changed or blocked")
    sys.exit(1)

rows = [r for r in table.select("tr") if len(r.select("td")) >= 5]
print(f"Parseable data rows in table.table-staples: {len(rows)}")

if rows:
    first = rows[0].select("td")
    print(
        f"First row: rank={first[0].get_text(strip=True)!r}  "
        f"card={first[1].get_text(strip=True)!r}  "
        f"pct={first[3].get_text(strip=True)!r}"
    )
    last = rows[-1].select("td")
    print(
        f"Last row:  rank={last[0].get_text(strip=True)!r}  "
        f"card={last[1].get_text(strip=True)!r}  "
        f"pct={last[3].get_text(strip=True)!r}"
    )

print()
if len(rows) <= 10:
    print("CONCLUSION: Only ~10 rows returned — full list is likely JS-rendered.")
    print(f"Inspect {OUT} to confirm (look for a <script> tag loading card data).")
else:
    print(
        f"CONCLUSION: {len(rows)} rows returned — page is static, pipeline gets the full list."
    )
