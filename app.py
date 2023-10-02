import sys

from src.logger import logging
from src.exception import CustomException

if __name__ == "__main__":
    logging.info("The execution has started")

    try:
        a = 1/0

    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e,sys)