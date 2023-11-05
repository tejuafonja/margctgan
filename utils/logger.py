import logging
import os

from datetime import datetime

today = datetime.today().date()

logdir = ".logs"
os.makedirs(logdir, exist_ok=True)


def get_logger(__name__):

    LOGGER = logging.getLogger(__name__)
    LOGGER.setLevel(logging.INFO)

    # Console handler.
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # File handler.
    file_handler = logging.FileHandler(f"{logdir}/{today}.log")
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s-%(levelname)s-[%(name)s]: %(message)s", datefmt="%I:%M:%S%p"
    )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    LOGGER.addHandler(console_handler)
    LOGGER.addHandler(file_handler)

    return LOGGER
