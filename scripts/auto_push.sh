#!/bin/zsh
set -e
cd /Users/bimalbairagya/Desktop/ClaimClaw
while true; do
  if [[ -n "$(git status --porcelain)" ]]; then
    git add -A
    git commit -m "auto: $(date '+%Y-%m-%d %H:%M:%S')" || true
    git push origin main || true
  fi
  sleep 30
done
