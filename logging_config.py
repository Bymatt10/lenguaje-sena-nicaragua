import logging
import os
from logging.handlers import RotatingFileHandler

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
SECURITY_LOG = os.path.join(LOG_DIR, 'security.log')


def configure_logging(app):
    os.makedirs(LOG_DIR, exist_ok=True)

    fmt = logging.Formatter(
        '%(asctime)s %(levelname)s %(name)s %(message)s'
    )

    sec_handler = RotatingFileHandler(
        SECURITY_LOG, maxBytes=2 * 1024 * 1024, backupCount=3, encoding='utf-8'
    )
    sec_handler.setLevel(logging.INFO)
    sec_handler.setFormatter(fmt)

    app.logger.setLevel(logging.INFO)
    app.logger.addHandler(sec_handler)
    app.logger.info('logging configured')