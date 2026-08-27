"""
Logging configuration for RepoCrawl.

Keeps logging setup centralized so every module can just do:

    import logging
    logger = logging.getLogger(__name__)

and get consistent formatting/behavior.
"""

import logging
import sys

from app.core.config import get_settings

_CONFIGURED = False


def configure_logging() -> None:
    """Configure root logging handlers. Safe to call multiple times."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    settings = get_settings()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(settings.log_level.upper())
    root_logger.handlers = [handler]

    # Quiet down noisy third-party loggers by default.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("git").setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Convenience accessor that guarantees logging has been configured."""
    configure_logging()
    return logging.getLogger(name)
