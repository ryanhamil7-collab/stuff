from loguru import logger
import sys
from pathlib import Path

def setup_logger(log_file: str = None, level: str = "INFO"):
    if log_file is None:
        log_file = Path(__file__).parent.parent.parent / "logs" / "trading_system.log"
    
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    logger.remove()
    
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True
    )
    
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=level,
        rotation="100 MB",
        retention="10 days",
        compression="zip"
    )
    
    return logger

log = setup_logger()
