#!/usr/bin/env bash
set -euo pipefail

SERVICE_OPENAI="CLAIMCLAW_OPENAI_API_KEY"
SERVICE_TWILIO="CLAIMCLAW_TWILIO_AUTH_TOKEN"

if ! command -v security >/dev/null 2>&1; then
  echo "error: macOS 'security' command not found."
  exit 1
fi

read -r -s -p "Enter OpenAI API key: " OPENAI_KEY
echo
if [[ -z "${OPENAI_KEY}" ]]; then
  echo "error: OpenAI key cannot be empty."
  exit 1
fi

read -r -s -p "Enter Twilio auth token (optional, press Enter to skip): " TWILIO_TOKEN
echo

security add-generic-password -a "$USER" -s "$SERVICE_OPENAI" -w "$OPENAI_KEY" -U >/dev/null

if [[ -n "${TWILIO_TOKEN}" ]]; then
  security add-generic-password -a "$USER" -s "$SERVICE_TWILIO" -w "$TWILIO_TOKEN" -U >/dev/null
fi

echo "Saved secrets to macOS Keychain:"
echo "- $SERVICE_OPENAI"
if [[ -n "${TWILIO_TOKEN}" ]]; then
  echo "- $SERVICE_TWILIO"
fi
