import os
from pathlib import Path
import logging
from logging.handlers import TimedRotatingFileHandler
import config

def get_current_directory():
    """
    Returns the current directory, either from the __file__ variable (if the script is being run from a file)
    or from os.getcwd() (if the script is being run from an interactive shell).

    Returns:
        pathlib.Path: The current directory.
    """
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path(os.getcwd())

def create_logger(log_file: str) -> logging.Logger:
    """
    Creates and configures a logger instance with a TimedRotatingFileHandler.

    Parameters:
    - log_file (str): The name of the log file.

    Returns:
    - logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger(log_file)
    if config.LOG_LEVEL == 'DEBUG':
        level = logging.DEBUG
    elif config.LOG_LEVEL == 'INFO':
        level = logging.INFO
    elif config.LOG_LEVEL == 'WARNING':
        level = logging.WARNING
    elif config.LOG_LEVEL == 'ERROR':
        level = logging.ERROR
    elif config.LOG_LEVEL == 'CRITICAL':
        level = logging.CRITICAL
    else:
        level = logging.INFO    
    logger.setLevel(level)
    
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    handler = TimedRotatingFileHandler(filename=f'{get_current_directory()}{os.sep}{config.PATH_LOGS}{os.sep}{log_file}', when='midnight', backupCount=config.LOG_BACKUPS)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger