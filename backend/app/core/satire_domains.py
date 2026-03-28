"""
backend/app/core/satire_domains.py

Domain blocklist for known satire/parody news sites.
Used in the fetch-url endpoint to flag articles before running the ML model.

The Onion and similar sites write in clean journalistic prose, which causes
TF-IDF and even DistilBERT to classify them as Real. A domain blocklist is
the most reliable fix since the writing style is intentionally indistinguishable
from real news.
"""

SATIRE_DOMAINS = {
    # Well-known satire sites
    "theonion.com",
    "babylonbee.com",
    "clickhole.com",           # Onion spinoff
    "thedailymash.co.uk",
    "newsthump.com",
    "thespoof.com",
    "waterfordwhispersnews.com",
    "reductress.com",
    "gomerblog.com",           # Medical satire
    "empirenews.net",
    "nationalreport.net",
    "worldnewsdailyreport.com",
    "huzlers.com",
    "thelapine.ca",
    "thebeaverton.com",        # Canadian satire
    "thepoke.co.uk",
    "newslo.com",
    "satira.news",
    "theshovel.com.au",        # Australian satire
    "dailycurrant.com",
    "literallyunbelievable.org",
    "newsbreaker.org",
}


def is_satire_domain(url: str) -> bool:
    """
    Returns True if the URL belongs to a known satire domain.
    Strips www. prefix for matching.
    """
    try:
        from urllib.parse import urlparse
        hostname = urlparse(url).hostname or ""
        hostname = hostname.lower().removeprefix("www.")
        return hostname in SATIRE_DOMAINS
    except Exception:
        return False
