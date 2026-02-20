#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/Users/bimalbairagya/Desktop/ClaimClaw"
DEFAULT_BRANCH="main"

cd "$PROJECT_DIR"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git init
fi

if [[ "$(git rev-parse --abbrev-ref HEAD)" != "$DEFAULT_BRANCH" ]]; then
  git branch -M "$DEFAULT_BRANCH"
fi

if ! git remote get-url origin >/dev/null 2>&1; then
  echo "No origin remote set. Example:"
  echo "  git remote add origin https://github.com/<user>/<repo>.git"
  exit 1
fi

git config --global credential.helper osxkeychain

echo "Git bootstrap complete."
echo "Origin: $(git remote get-url origin)"
