"""CLI script for downloading SEC filings.

Usage:
    python scripts/ingest.py --ticker AAPL --filing-type 10-K --count 1
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add src to path for direct script execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import structlog

from finrag.config import get_settings
from finrag.ingestion.edgar_client import (
    EdgarError,
    ingest_filing,
)

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ],
)

logger = structlog.get_logger(__name__)


def main() -> None:
    """Parse CLI args and run filing ingestion."""
    parser = argparse.ArgumentParser(
        description="Download SEC filings from EDGAR",
    )
    parser.add_argument(
        "--ticker",
        required=True,
        help="Stock ticker symbol (e.g., AAPL)",
    )
    parser.add_argument(
        "--filing-type",
        required=True,
        help="Filing type (e.g., 10-K, 10-Q, 8-K)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of recent filings to download (default: 1)",
    )

    args = parser.parse_args()

    settings = get_settings()

    logger.info(
        "ingestion_start",
        ticker=args.ticker,
        filing_type=args.filing_type,
        count=args.count,
    )

    try:
        saved_paths = asyncio.run(
            ingest_filing(
                ticker=args.ticker,
                filing_type=args.filing_type,
                settings=settings,
                count=args.count,
            )
        )

        for path in saved_paths:
            logger.info("filing_saved", path=str(path))

        logger.info(
            "ingestion_complete",
            total_saved=len(saved_paths),
        )

    except EdgarError as e:
        logger.error("ingestion_failed", error=str(e))
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("ingestion_interrupted")
        sys.exit(130)


if __name__ == "__main__":
    main()
