# Contributing to FinRAG

This project is built over 15 days using an AI-guided mentorship workflow. Each day produces a runnable, testable, committable increment.

---

## How the Build Works

Each day is guided by an AI mentor (Claude, Gemini, etc.) using the system prompt in `RULES.md`. The mentor asks conceptual questions before giving code, flags production vs demo trade-offs, and ensures each day ends with working, tested software.

### For New Team Members

1. **Clone and set up:**
   ```bash
   git clone https://github.com/MetaFazer/Finrag.git
   cd Finrag
   python -m venv .venv
   .venv\Scripts\activate          # Windows
   # source .venv/bin/activate     # Mac/Linux
   pip install -e ".[dev]"
   cp .env.example .env
   ```

2. **Check current progress:**
   - Open `ROADMAP.md` to see which days are complete (✅) and what's next
   - Open `DEBT_LEDGER.md` to see known shortcuts and TODOs

3. **Start your AI mentor session:**
   - Open your AI tool (Claude Code, Gemini in IDE, etc.)
   - Paste the entire contents of `RULES.md` as your first message
   - The AI will respond with the project overview and roadmap
   - Tell it which day you're on, e.g.: `"Day 1 is complete. Proceed to Day 2."`
   - The AI will expand Day 2 with objectives, concepts, tasks, and tests

4. **Work through the day:**
   - Answer the conceptual questions the mentor asks
   - Implement the tasks
   - Run the tests
   - Check off the verification checklist

5. **When the day is done:**
   - All tests pass
   - Verification checklist is complete
   - Update `ROADMAP.md` status column (change ⬜ to ✅)
   - Commit and push
   - Type: `"Day N complete. Confirmed. Proceed to Day N+1."`

---

## Day Progression Rules

- **Never skip days.** Each day builds on the previous one.
- **Never proceed without passing tests.** If tests fail, fix them first.
- **Update ROADMAP.md** when you complete a day so the team knows where things stand.
- **Update DEBT_LEDGER.md** if you take any shortcuts that need future resolution.

---

## Code Standards

These are enforced from Day 1:

- Every function has type hints and a docstring (Args, Returns, Raises)
- No magic numbers. Constants are named and explained.
- Errors are caught at the layer that can handle them, not silently swallowed.
- Structured logging (structlog, JSON format) after Day 3. No bare `print()`.
- No hardcoded API keys. Environment variables only.
- Every module has a corresponding test file on the day it's created.

---

## Running Tests

```bash
# All tests
python -m pytest tests/ -v

# Unit tests only (no network)
python -m pytest tests/ -v -k "not Integration"

# Integration tests (hits SEC EDGAR)
python -m pytest tests/ -v -k "Integration"

# With coverage
python -m pytest tests/ -v --cov=finrag
```

---

## Project Structure

See `ROADMAP.md` for the full target directory structure with day annotations showing when each file gets created.
