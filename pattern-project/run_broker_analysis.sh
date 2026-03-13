# Convenience script to run programs within the brokerage intelligence module.

# Get the absolute path of the project root
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
BIN_DIR="$PROJECT_ROOT/code/broker_analysis"
DATA_DIR="$PROJECT_ROOT/data/broker_analysis"
RESULTS_DIR="$PROJECT_ROOT/results/broker_analysis"
VENV_ACTIVATE="$PROJECT_ROOT/venv_broker/bin/activate"

# Check if environment exists
if [ ! -f "$VENV_ACTIVATE" ]; then
    echo "❌ Error: Virtual environment not found at $VENV_ACTIVATE"
    exit 1
fi

# Source the environment using POSIX compliant dot command
. "$VENV_ACTIVATE"

# If no arguments provided, show help
if [ $# -eq 0 ]; then
    echo "🚀 Brokerage Intelligence Tool Wrapper"
    echo "Usage: ./run_broker_analysis.sh <script_name.py> [args...]"
    echo ""
    echo "Available scripts in code/broker_analysis/:"
    ls -p "$BIN_DIR" | grep -v / | grep .py
    exit 0
fi

# Run the requested script
SCRIPT_NAME=$1
shift # Remove script name from arguments

if [ -f "$BIN_DIR/$SCRIPT_NAME" ]; then
    echo "🏃 Running $SCRIPT_NAME..."
    # We change directory to code/broker_analysis so the script can find its relative config/classes
    # But we make sure it knows where to put reports if it has a setting for it.
    cd "$BIN_DIR"
    python3 "$SCRIPT_NAME" "$@"
else
    echo "❌ Error: Script '$SCRIPT_NAME' not found in $BIN_DIR"
    exit 1
fi
