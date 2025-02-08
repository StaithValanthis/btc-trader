import os
import logging
import structlog
from structlog.dev import ConsoleRenderer

def configure_logger():
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    debug_mode = os.getenv("DEBUG_MODE", "false").lower() == "true"

    # Common processors for all environments
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    # Use different renderers based on DEBUG_MODE
    if debug_mode:
        # Human-readable output with Unicode support
        processors.append(ConsoleRenderer(colors=True))
    else:
        # Structured JSON for production
        processors.append(structlog.processors.JSONRenderer())

    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configure standard logging
    logging.basicConfig(
        format="%(message)s" if debug_mode else None,
        level=log_level,
        handlers=[logging.StreamHandler()]
    )

# Initialize the logger
configure_logger()
logger = structlog.get_logger()

def progress_bar(percentage, bar_length=40):
    """Generate a terminal-friendly progress bar."""
    filled_length = int(bar_length * percentage / 100)
    bar = "█" * filled_length + "-" * (bar_length - filled_length)

    # Force UTF-8 encoding for terminal display
    return f"[{bar}] {percentage:.1f}%"

