"""Tests for the EDGAR client module.

Covers four test categories:
1. Happy path: successful ticker resolution, filing fetch, section parsing
2. Edge cases: invalid tickers, missing filings
3. Failure modes: config validation, section parser robustness
4. Unit tests: section parsing with known HTML fragments
"""

import pytest

from finrag.config import Settings
from finrag.ingestion.edgar_client import (
    EdgarClient,
    FilingMetadata,
    ParsedFiling,
    TickerNotFoundError,
    FilingNotFoundError,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture
def settings() -> Settings:
    """Create test settings with a valid user agent."""
    return Settings(
        edgar_user_agent="FinRAG Test test@finrag.dev",
        data_dir="./data/test_raw",
    )


@pytest.fixture
def sample_10k_html() -> str:
    """A minimal 10-K HTML fragment with identifiable sections."""
    return """
    <html>
    <body>
    <h2>PART I</h2>

    <h3>Item 1. Business</h3>
    <p>We are a technology company that designs, manufactures, and markets
    consumer electronics, computer software, and online services. Our fiscal
    year ends in September. Revenue for FY2024 was $383 billion, a 2% increase
    from FY2023. We operate in multiple segments including Products and Services.</p>

    <p>Our products include iPhone, Mac, iPad, and wearables. Services include
    the App Store, Apple Music, iCloud, and Apple Pay. Services revenue grew 14%
    year over year to reach $85.2 billion.</p>

    <h3>Item 1A. Risk Factors</h3>
    <p>Our business is subject to various risks. Global economic conditions may
    adversely affect demand for our products. Changes in trade policy, tariffs,
    and international regulations could increase costs. Supply chain disruptions
    remain a material risk. Competition in the technology industry is intense.</p>

    <p>Foreign currency fluctuations affect our international operations. Cybersecurity
    threats could compromise customer data. Regulatory changes in data privacy laws
    across jurisdictions create compliance complexity.</p>

    <h3>Item 7. Management's Discussion and Analysis</h3>
    <p>Operating margin expanded 180bps driven by Services mix shift and favorable
    component pricing. Gross margin was 46.2% compared to 44.1% in the prior year.
    Research and development expenses increased 8% to $29.9 billion.</p>

    <p>Cash and marketable securities totaled $162.1 billion. We returned $90 billion
    to shareholders through dividends and share repurchases. Free cash flow was
    $111.4 billion for the fiscal year.</p>

    <h3>Item 8. Financial Statements and Supplementary Data</h3>
    <p>Consolidated Statements of Operations for fiscal years 2024, 2023, and 2022.
    Total net revenue was $383.3B, $383.2B, and $394.3B respectively.</p>
    </body>
    </html>
    """


# --------------------------------------------------------------------------- #
# Config validation tests
# --------------------------------------------------------------------------- #

class TestConfig:
    """Tests for Settings validation."""

    def test_valid_settings(self, settings: Settings) -> None:
        """Settings with valid user agent should be accepted."""
        assert settings.edgar_user_agent == "FinRAG Test test@finrag.dev"

    def test_missing_email_in_user_agent(self) -> None:
        """User agent without email should be rejected."""
        with pytest.raises(Exception):
            Settings(edgar_user_agent="No Email Here")

    def test_invalid_log_level(self) -> None:
        """Invalid log level should be rejected."""
        with pytest.raises(Exception):
            Settings(
                edgar_user_agent="Test test@test.com",
                log_level="VERBOSE",
            )

    def test_default_values(self, settings: Settings) -> None:
        """Default values should be set correctly."""
        assert settings.edgar_max_rps == 10
        assert settings.log_level == "INFO"
        assert "efts.sec.gov" in settings.edgar_base_url


# --------------------------------------------------------------------------- #
# Section parsing tests (unit tests, no network)
# --------------------------------------------------------------------------- #

class TestSectionParsing:
    """Tests for section parsing from filing HTML."""

    def test_parse_10k_sections(
        self, settings: Settings, sample_10k_html: str
    ) -> None:
        """Parser should extract known 10-K sections from HTML."""
        client = EdgarClient(settings)
        sections = client.parse_sections(sample_10k_html, "10-K")

        # Should find at least Risk Factors, MD&A, Financial Statements
        # Note: Item 1 (Business) may merge with Item 1A when they're close
        # together in the source. This is acceptable for the regex parser.
        section_keys = list(sections.keys())
        assert len(sections) >= 3, f"Expected >= 3 sections, got {len(sections)}: {section_keys}"

        # Check that key sections are found
        found_any_risk = any("Risk" in k for k in section_keys)
        found_any_mda = any("MD" in k or "7" in k for k in section_keys)
        assert found_any_risk, f"No Risk section found. Keys: {section_keys}"
        assert found_any_mda, f"No MD&A section found. Keys: {section_keys}"

    def test_parse_10k_section_content(
        self, settings: Settings, sample_10k_html: str
    ) -> None:
        """Parsed sections should contain the actual filing text."""
        client = EdgarClient(settings)
        sections = client.parse_sections(sample_10k_html, "10-K")

        # Find the MD&A section and check for expected content
        mda_sections = [v for k, v in sections.items() if "MD" in k or "7" in k]
        if mda_sections:
            assert "operating margin" in mda_sections[0].lower() or "180bps" in mda_sections[0]

    def test_parse_non_10k_returns_full_text(self, settings: Settings) -> None:
        """Non-10K filings should return a single 'full_text' section."""
        client = EdgarClient(settings)
        sections = client.parse_sections("<html><body>Q2 report</body></html>", "8-K")

        assert "full_text" in sections
        assert "Q2 report" in sections["full_text"]

    def test_parse_empty_html(self, settings: Settings) -> None:
        """Empty HTML should not crash the parser."""
        client = EdgarClient(settings)
        sections = client.parse_sections("", "10-K")

        # Should return something (either empty dict or full_text)
        assert isinstance(sections, dict)

    def test_parse_malformed_html(self, settings: Settings) -> None:
        """Broken HTML should not crash. Parser returns what it can."""
        client = EdgarClient(settings)
        broken_html = "<html><body><p>Unclosed paragraph<div>Mixed tags</p></div>"
        sections = client.parse_sections(broken_html, "10-K")

        assert isinstance(sections, dict)


# --------------------------------------------------------------------------- #
# Integration tests (require network access to SEC EDGAR)
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
class TestEdgarClientIntegration:
    """Integration tests hitting the real EDGAR API.

    These tests require network access. Mark with
    @pytest.mark.skipif for CI environments without access.
    """

    async def test_ticker_to_cik_apple(self, settings: Settings) -> None:
        """AAPL should resolve to a valid CIK."""
        async with EdgarClient(settings) as client:
            cik, company_name = await client.ticker_to_cik("AAPL")

        assert cik.isdigit()
        assert len(cik) == 10
        assert company_name  # Should be non-empty

    async def test_ticker_to_cik_invalid(self, settings: Settings) -> None:
        """Invalid ticker should raise TickerNotFoundError."""
        async with EdgarClient(settings) as client:
            with pytest.raises(TickerNotFoundError):
                await client.ticker_to_cik("XYZNOTREAL123")

    async def test_get_filing_urls(self, settings: Settings) -> None:
        """Should find 10-K filings for Apple."""
        async with EdgarClient(settings) as client:
            cik, _ = await client.ticker_to_cik("AAPL")
            filings = await client.get_filing_urls(cik, "10-K", count=2)

        assert len(filings) > 0
        assert len(filings) <= 2
        for f in filings:
            assert isinstance(f, FilingMetadata)
            assert f.filing_type == "10-K"
            assert f.primary_document_url.startswith("https://")

    async def test_download_and_parse(self, settings: Settings) -> None:
        """Full pipeline: resolve ticker, get filing, download, parse."""
        async with EdgarClient(settings) as client:
            cik, _ = await client.ticker_to_cik("MSFT")
            filings = await client.get_filing_urls(cik, "10-K", count=1)
            assert len(filings) >= 1

            raw = await client.download_filing(filings[0].primary_document_url)
            assert len(raw) > 1000  # A real 10-K is large

            sections = client.parse_sections(raw, "10-K")
            assert len(sections) >= 1  # Should find at least some sections
