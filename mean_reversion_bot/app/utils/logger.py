import structlog, logging, os

def configure():
    lvl = os.getenv("LOG_LEVEL","INFO").upper()
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger
    )
    logging.basicConfig(level=lvl,format="%(message)s")

configure()
logger = structlog.get_logger()
