#!/usr/bin/env bash
set -euo pipefail

SERVICE_OPENAI="CLAIMCLAW_OPENAI_API_KEY"
SERVICE_TWILIO="CLAIMCLAW_TWILIO_AUTH_TOKEN"

if ! command -v security >/dev/null 2>&1; then
  echo "error: macOS 'security' command not found."
  return 1 2>/dev/null || exit 1
fi

OPENAI_KEY="$(security find-generic-password -a "$USER" -s "$SERVICE_OPENAI" -w 2>/dev/null || true)"
if [[ -z "${OPENAI_KEY}" ]]; then
  echo "error: missing $SERVICE_OPENAI in Keychain. Run scripts/set-secrets.sh first."
  return 1 2>/dev/null || exit 1
fi

export OPENAI_API_KEY="$OPENAI_KEY"
export LLM_PROVIDER="${LLM_PROVIDER:-openai}"
export OPENAI_CHAT_MODEL="${OPENAI_CHAT_MODEL:-gpt-4o}"
export OPENAI_VISION_MODEL="${OPENAI_VISION_MODEL:-gpt-4o}"

TWILIO_TOKEN="$(security find-generic-password -a "$USER" -s "$SERVICE_TWILIO" -w 2>/dev/null || true)"
if [[ -n "${TWILIO_TOKEN}" ]]; then
  export TWILIO_AUTH_TOKEN="$TWILIO_TOKEN"
fi

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  . .venv/bin/activate
fi

echo "ClaimClaw secrets loaded into current shell."
echo "OPENAI_API_KEY=loaded"
if [[ -n "${TWILIO_TOKEN}" ]]; then
  echo "TWILIO_AUTH_TOKEN=loaded"
fi
