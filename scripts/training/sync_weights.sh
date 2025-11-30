#!/bin/bash
# Script to sync model weights between training (GPU) and playing (laptop) machines
# Source code is synced via git, so this script only syncs the models/ directory

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Default values
REMOTE_HOST=""
REMOTE_USER=""
REMOTE_PATH=""
LOCAL_PATH="${PROJECT_DIR}/models"
DIRECTION="pull"  # pull: from remote to local, push: from local to remote
DRY_RUN=false
EXCLUDE_PATTERNS=(
    "__pycache__"
    "*.pyc"
    "*.pyo"
    ".DS_Store"
    "Thumbs.db"
    ".git"
    "*.log"
)

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Sync model weights between GPU training machine and laptop.

Options:
    -h, --host HOST          Remote hostname or IP (required)
    -u, --user USER          Remote username (optional, uses current user if not specified)
    -p, --path PATH          Remote path to project root (default: same as local)
    -d, --direction DIR      Sync direction: 'pull' (from remote to local) or 'push' (from local to remote)
                             Default: pull
    -l, --local-path PATH    Local models directory path (default: ./models)
    -n, --dry-run            Show what would be synced without actually syncing
    --help                   Show this help message

Examples:
    # Pull weights from GPU machine to laptop
    $0 -h gpu-machine.example.com -u myuser

    # Push weights from laptop to GPU machine
    $0 -h gpu-machine.example.com -u myuser -d push

    # Use SSH config host alias
    $0 -h gpu-box -d pull

    # Dry run to see what would be synced
    $0 -h gpu-machine.example.com -n

Notes:
    - Source code should be synced via git
    - Only syncs the models/ directory
    - Excludes cache files, logs, and git directories
    - Uses rsync with compression and progress display
EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--host)
            REMOTE_HOST="$2"
            shift 2
            ;;
        -u|--user)
            REMOTE_USER="$2"
            shift 2
            ;;
        -p|--path)
            REMOTE_PATH="$2"
            shift 2
            ;;
        -d|--direction)
            DIRECTION="$2"
            shift 2
            ;;
        -l|--local-path)
            LOCAL_PATH="$2"
            shift 2
            ;;
        -n|--dry-run)
            DRY_RUN=true
            shift
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "${REMOTE_HOST}" ]]; then
    echo "Error: Remote host is required"
    echo ""
    usage
    exit 1
fi

# Validate direction
if [[ "${DIRECTION}" != "pull" && "${DIRECTION}" != "push" ]]; then
    echo "Error: Direction must be 'pull' or 'push'"
    exit 1
fi

# Build remote path
if [[ -z "${REMOTE_PATH}" ]]; then
    # Try to detect remote path from git config or use default
    REMOTE_PATH="${PROJECT_DIR}"
fi

# Build remote specification
if [[ -n "${REMOTE_USER}" ]]; then
    REMOTE_SPEC="${REMOTE_USER}@${REMOTE_HOST}"
else
    REMOTE_SPEC="${REMOTE_HOST}"
fi

REMOTE_MODELS_PATH="${REMOTE_SPEC}:${REMOTE_PATH}/models"

# Ensure local models directory exists
mkdir -p "${LOCAL_PATH}"

# Build exclude options
EXCLUDE_OPTS=()
for pattern in "${EXCLUDE_PATTERNS[@]}"; do
    EXCLUDE_OPTS+=("--exclude=${pattern}")
done

# Build rsync command
RSYNC_OPTS=(
    -avz                    # archive, verbose, compress
    --progress              # show progress
    --human-readable        # human-readable sizes
    "${EXCLUDE_OPTS[@]}"   # exclude patterns
)

if [[ "${DRY_RUN}" == true ]]; then
    RSYNC_OPTS+=("--dry-run")
fi

# Determine source and destination based on direction
if [[ "${DIRECTION}" == "pull" ]]; then
    SOURCE="${REMOTE_MODELS_PATH}/"
    DEST="${LOCAL_PATH}/"
    echo "Pulling model weights from ${REMOTE_SPEC}..."
else
    SOURCE="${LOCAL_PATH}/"
    DEST="${REMOTE_MODELS_PATH}/"
    echo "Pushing model weights to ${REMOTE_SPEC}..."
fi

if [[ "${DRY_RUN}" == true ]]; then
    echo "DRY RUN - No files will be modified"
    echo ""
fi

# Run rsync
echo "Source: ${SOURCE}"
echo "Destination: ${DEST}"
echo ""

rsync "${RSYNC_OPTS[@]}" "${SOURCE}" "${DEST}"

if [[ "${DRY_RUN}" != true ]]; then
    echo ""
    echo "Sync completed successfully!"
    echo ""
    echo "Synced models directory: ${LOCAL_PATH}"
    if [[ "${DIRECTION}" == "pull" ]]; then
        echo "Weights pulled from: ${REMOTE_SPEC}"
    else
        echo "Weights pushed to: ${REMOTE_SPEC}"
    fi
fi

