#!/usr/bin/env bash
set -euo pipefail
grep -RIn --line-number -E "now_utc\s*:" csp || true
