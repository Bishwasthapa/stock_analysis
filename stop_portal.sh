#!/bin/bash
# Stop the Nepal Stock Pattern Hub server
if fuser 8000/tcp > /dev/null 2>&1; then
    fuser -k 8000/tcp
    echo "✅ Portal stopped."
else
    echo "ℹ️  Portal is not running."
fi
