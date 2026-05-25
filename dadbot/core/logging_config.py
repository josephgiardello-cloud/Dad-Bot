"""
Structured logging configuration for DadBot.
Sets up logging handlers, formatters, and log levels for observability.
"""
import logging
import sys

def configure_logging(level=logging.INFO):
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt='%(asctime)s %(levelname)s %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
    logging.getLogger('dadbot').setLevel(level)
