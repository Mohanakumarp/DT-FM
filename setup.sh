#!/usr/bin/env bash
# Convenience wrapper so a new clone can run: bash setup.sh
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/setup_new_system.sh" "$@"
