#!/bin/bash
# Generate test audio samples using macOS 'say' command.
# Outputs 16 kHz mono WAV files compatible with both Whisper and VibeVoice.
# Requires: macOS say, ffmpeg

set -e
SAMPLES_DIR="$(cd "$(dirname "$0")/.." && pwd)/data/samples"
mkdir -p "$SAMPLES_DIR"

generate() {
    local name="$1"
    local text="$2"
    local aiff="$SAMPLES_DIR/${name}.aiff"
    local wav="$SAMPLES_DIR/${name}.wav"

    echo "Generating: $name"
    say -o "$aiff" "$text"

    if command -v ffmpeg &>/dev/null; then
        ffmpeg -y -i "$aiff" -ar 16000 -ac 1 "$wav" -loglevel error
        rm "$aiff"
        echo "  Saved: $wav"
    else
        echo "  ffmpeg not found. Install with: brew install ffmpeg"
        echo "  Kept AIFF: $aiff"
    fi
}

generate "sample_clean" \
    "The patient presents with acute chest pain radiating to the left arm, onset two hours ago."

generate "sample_medical" \
    "Diagnosis: hypertension with comorbid type two diabetes mellitus. Prescribed metformin five hundred milligrams twice daily."

generate "sample_noisy" \
    "Please schedule a follow-up appointment for next Tuesday at three pm with Doctor Johnson."

echo "Done."
