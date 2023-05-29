"""log"""

import os
import functools
import logging
import logging.handlers

from typing import Optional

current_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(current_dir, "../logs/knowledgeable-prompt-tuning.log")

log_level = os.getenv("LOG_LEVEL", "DEBUG")

class Logger(logging.Logger):
    def __init__(self, name: Optional[str] = None):
        self.logger = logging.getLogger(name)
        for name in ['boto', 'urllib3', 's3transfer', 'boto3', 'botocore', 'nose']:
            logging.getLogger(name).setLevel(logging.CRITICAL)

        log_config = {
            'DEBUG': 10,
            'INFO': 20,
            'TRAIN': 21,
            'EVAL': 22,
            'WARNING': 30,
            'ERROR': 40,
            'CRITICAL': 50,
            'EXCEPTION': 100,
        }
        for key, level in log_config.items():
            logging.addLevelName(level, key)
            if key == 'EXCEPTION':
                self.__dict__[key.lower()] = self.logger.exception
            else:
                self.__dict__[key.lower()] = functools.partial(self.__call__,
                                                               level)

        self.format = logging.Formatter(
            fmt='[%(asctime)-15s] [%(levelname)8s] - %(message)s')

        self.stream_handler = logging.StreamHandler()
        self.stream_handler.setFormatter(self.format)
        log_dir = os.path.dirname(os.path.abspath(log_file))
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.file_handler = logging.handlers.RotatingFileHandler(
            log_file, mode="a", maxBytes=10*1024*1024,
            backupCount=2)
        self.file_handler.setFormatter(self.format)

        self.logger.addHandler(self.stream_handler)
        self.logger.addHandler(self.file_handler)
        self.logger.setLevel(getattr(logging, log_level))
        self.logger.propagate = False

    def __call__(self, log_level: int, msg: str):
        self.logger.log(log_level, msg)


logger = Logger()
