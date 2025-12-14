import logging
import sys
import os
from datetime import datetime
from pathlib import Path

def is_docker_environment():
    """Check if running inside Docker container."""
    return os.path.exists('/.dockerenv') or os.environ.get('DOCKER_ENV') == 'true'


def setup_logger(name=__name__, log_dir=None):
    """
    Sets up a logger with environment-aware output:
    - Local dev: Console only
    - Docker: Console + file logging
    
    Args:
        name: Logger name (usually __name__)
        log_dir: Optional specific log directory for timestamped runs (Docker only)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] - %(message)s')
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if is_docker_environment():
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(
                os.path.join(log_dir, 'run.log'),
                encoding='utf-8'
            )
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        general_log_dir = Path('logs')
        general_log_dir.mkdir(exist_ok=True)
        general_handler = logging.FileHandler(
            general_log_dir / 'run.log',
            encoding='utf-8',
            mode='a'
        )
        general_handler.setLevel(logging.INFO)
        general_handler.setFormatter(formatter)
        logger.addHandler(general_handler)
    
    return logger


def get_timestamp():
    """Generate timestamp string for run identification."""
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def load_config():
    pass
