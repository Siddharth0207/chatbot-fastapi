"""Logging configuration module for the application.

Sets up centralized logging with timestamped log files. Creates logs in a dedicated
logs directory with a timestamp-based filename for easy organization and tracking.
"""

import logging
import os
from datetime import datetime

# Generate timestamped log filename
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
log_path = os.path.join(os.getcwd(), "logs", LOG_FILE)
os.makedirs(log_path, exist_ok=True)

# Full path to the log file
LOG_FILE_PATH = os.path.join(log_path, LOG_FILE)

# Configure logging with timestamp, line number, logger name, level, and message
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)