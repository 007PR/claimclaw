#!/bin/zsh
set -u

PROJECT_DIR="/Users/bimalbairagya/Desktop/ClaimClaw"
PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"

cd "$PROJECT_DIR" || exit 1

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

echo "[$(timestamp)] auto-push watcher started (tests required before push)"

while true; do
  if [[ -n "$(git status --porcelain)" ]]; then
    echo "[$(timestamp)] changes detected; staging and running tests"
    git add -A

    if git diff --cached --quiet; then
      sleep 30
      continue
    fi

    if "$PYTHON_BIN" -m pytest -q -p no:cacheprovider; then
      COMMIT_MSG="auto: $(timestamp)"
      if git commit -m "$COMMIT_MSG"; then
        if git push origin main; then
          echo "[$(timestamp)] pushed successfully"
        else
          echo "[$(timestamp)] push failed; keeping commit locally"
        fi
      else
        echo "[$(timestamp)] commit skipped (no staged delta)"
      fi
    else
      echo "[$(timestamp)] tests failed; skipping commit/push"
      # Unstage so user can inspect/edit normally; working changes are preserved.
      git reset >/dev/null 2>&1 || true
    fi
  fi

  sleep 30
done
