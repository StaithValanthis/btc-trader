# File: app/utils/logger.py

import structlog
import logging
import os

def configure_logger():
    # Default to INFO level, can override with LOG_LEVEL=DEBUG if needed
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    # Instead of structlog.processors.JSONRenderer, we can do KeyValueRenderer
    # or a simpler JSON with fewer fields:
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            # For minimal text logs:
            # structlog.processors.KeyValueRenderer(key_order=["timestamp","level","event"]),
            # If you still want JSON but fewer fields:
            structlog.processors.JSONRenderer(
                ensure_ascii=False,
                # override any config if needed
            ),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # logging.basicConfig affects the stdlib root logger
    logging.basicConfig(
        format="%(message)s",
        level=log_level,
        handlers=[logging.StreamHandler()]
    )

configure_logger()
logger = structlog.get_logger()
