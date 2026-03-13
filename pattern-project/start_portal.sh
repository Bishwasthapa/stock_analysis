#!/bin/bash
# Start the Nepal Stock Pattern Hub (foreground — logs visible in terminal, Ctrl+C to stop)
cd "$(dirname "$0")"
export PYTHONPATH=.
echo "🚀 Starting Nepal Stock Pattern Hub at http://localhost:8000"
./venv/bin/uvicorn code.api.main:app --host 0.0.0.0 --port 8000 &> /tmp/portal_debug.log
