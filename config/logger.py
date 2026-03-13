import logging
import os
import sys
from config.config import LOG_LEVEL, LOG_FORMAT, LOG_DIR


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers on re-import
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    formatter = logging.Formatter(LOG_FORMAT)

    # ── Console Handler (UTF-8 safe for Windows) ─────
    if sys.platform == "win32":
        import io
        utf8_stream = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace"
        )
        console_handler = logging.StreamHandler(utf8_stream)
    else:
        console_handler = logging.StreamHandler(sys.stdout)

    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ── File Handler (always UTF-8) ───────────────────
    os.makedirs(LOG_DIR, exist_ok=True)
    file_handler = logging.FileHandler(
        os.path.join(LOG_DIR, "pipeline.log"), encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
