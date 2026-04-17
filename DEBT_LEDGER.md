# FinRAG Debt Ledger

Tracking shortcuts and TODOs across the 15-day build. Each item is tagged with the day it was introduced and the day it's scheduled for resolution.

## Active Debt

| ID | Day | Description | Resolution Target |
|----|-----|-------------|-------------------|
| DAY-1-001 | 1 | Section parser uses regex + BeautifulSoup heuristics, not a proper filing-type-specific parser | Day 2 |
| DAY-1-002 | 1 | No local CIK lookup cache, every resolution hits EDGAR API | Day 3 |
| DAY-1-003 | 1 | Ingestion script is sync wrapper around async code | Acceptable for CLI; async matters at API layer (Day 11) |

## Resolved Debt

| ID | Introduced | Resolved | Description |
|----|-----------|----------|-------------|
| | | | |
