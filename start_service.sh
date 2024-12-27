#!/bin/bash

# Set environment variables
export http_proxy=http://150.6.13.63:3128
export https_proxy=http://150.6.13.63:3128
export OPENAI_PROXY=http://150.6.13.63:3128
export no_proxy=swm-02-01
export ENVIRONMENT='product'

# Define the path to the log file
LOG_FILE="/var/webapps/nohup_backend.out"

# Check if the log file exists
if [ -f "$LOG_FILE" ]; then
    # If the file exists, create a backup with a timestamp
    BACKUP_FILE="/var/webapps/nohup_backend.out.bak-$(date +%Y%m%d-%H%M%S)"
    cp "$LOG_FILE" "$BACKUP_FILE"
    echo "Backup of $LOG_FILE created as $BACKUP_FILE"
else
    echo "$LOG_FILE does not exist. No backup needed."
fi

# Check if any uvicorn processes are running
PIDS=$(pgrep -f uvicorn)

if [ -n "$PIDS" ]; then
    # If uvicorn processes are found, kill all of them
    echo "Uvicorn processes are running. Killing all processes..."
    echo "$PIDS" | xargs kill
    echo "All running uvicorn processes killed."
else
    echo "No running uvicorn processes found."
fi

# Set the number of workers for the uvicorn service
WORKERS=1

# Start the uvicorn service in the background using nohup and the python command from venv
nohup python3.12 -m uvicorn app.main:app --host 0.0.0.0 --workers $WORKERS > "$LOG_FILE" 2>&1 &

echo "Uvicorn service started. Output being logged to $LOG_FILE"
