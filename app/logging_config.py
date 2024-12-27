import logging
import os
from logging.handlers import TimedRotatingFileHandler
from uvicorn.config import LOGGING_CONFIG

# Customize the format to include the time at the start
LOGGING_CONFIG["formatters"]["default"] = {
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "datefmt": "%Y-%m-%d %H:%M:%S"
}

# Update the access log format to include timestamps
LOGGING_CONFIG["formatters"]["access"] = {
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "datefmt": "%Y-%m-%d %H:%M:%S"
}
# Apply the changes to the logging system
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

logging.config.dictConfig(LOGGING_CONFIG)


# Fetch log directory from environment variables or use the default location
log_dir = os.getenv("LOG_DIR", "/var/webapps/logs/daisy_backend")

# Ensure the log directory exists
os.makedirs(log_dir, exist_ok=True)

# Configure the TimedRotatingFileHandler
log_file_handler = TimedRotatingFileHandler(
    filename=os.path.join(log_dir, "app.log"),  # Log file location
    when="midnight",  # Rotate at midnight
    interval=1,       # Rotate every day
    backupCount=0     # Keep the last 7 days of logs
)

# Set the log format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log_file_handler.setFormatter(formatter)

# Set the log level
log_file_handler.setLevel(logging.INFO)

# Configure console handler for terminal logs
# console_handler = logging.StreamHandler()
# console_handler.setFormatter(formatter)
# console_handler.setLevel(logging.INFO)

# Get the root logger and add both handlers
root_logger = logging.getLogger()
root_logger.addHandler(log_file_handler)
# root_logger.addHandler(console_handler)
root_logger.setLevel(logging.INFO)

# Configure the Uvicorn access logger to log only to the file, not the console
uvicorn_access_logger = logging.getLogger("uvicorn.access")
uvicorn_access_logger.addHandler(log_file_handler)
uvicorn_access_logger.setLevel(logging.INFO)
# Do not propagate to console
uvicorn_access_logger.propagate = False

# Configure the Uvicorn error logger (if needed for debugging)
uvicorn_error_logger = logging.getLogger("uvicorn.error")
uvicorn_error_logger.addHandler(log_file_handler)
# uvicorn_error_logger.addHandler(console_handler)
uvicorn_error_logger.setLevel(logging.INFO)

# Log startup info
root_logger.info("Logging is configured. Logs will be saved to: %s", log_dir)
