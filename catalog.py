#!/usr/bin/env python3
"""
Audio catalog script - analyzes audio files and outputs CSV rows to stdout.

Usage:
    catalog.py [--header] <file1> [file2] ...
    catalog.py --header sound/**/*.wav > catalog.csv
"""

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from google import genai
from mutagen.mp3 import MP3
from mutagen.wave import WAVE

CSV_COLUMNS = [
    "file_path",
    "file_name",
    "directory",
    "file_size_bytes",
    "format",
    "duration_seconds",
    "sample_rate",
    "channels",
    "bit_depth",
    "bitrate_kbps",
    "file_hash",
    "ai_description",
    "ai_tags",
    "ai_category",
    "processed_at",
]

VALID_CATEGORIES = {
    "speech", "creature", "weapon", "ambient", "music",
    "ui", "mechanical", "explosion", "footstep", "other"
}

ANALYSIS_PROMPT = """Analyze this audio file from the video game Half-Life.

Respond with ONLY a JSON object (no markdown, no extra text):
{"description": "1-2 sentence description", "tags": ["tag1", "tag2", "tag3"], "category": "category"}

category MUST be exactly one of: speech, creature, weapon, ambient, music, ui, mechanical, explosion, footstep, other

Example: {"description": "A scientist says hello", "tags": ["speech", "male", "greeting"], "category": "speech"}"""


def compute_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of file (first 16 chars)."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()[:16]


def extract_metadata(file_path: Path) -> dict:
    """Extract technical audio properties using mutagen."""
    stat = file_path.stat()
    suffix = file_path.suffix.lower()

    metadata = {
        "file_path": str(file_path),
        "file_name": file_path.name,
        "directory": file_path.parent.name,
        "file_size_bytes": stat.st_size,
        "format": suffix.lstrip("."),
        "duration_seconds": None,
        "sample_rate": None,
        "channels": None,
        "bit_depth": None,
        "bitrate_kbps": None,
        "file_hash": compute_hash(file_path),
    }

    if suffix == ".wav":
        audio = WAVE(file_path)
        info = audio.info
        metadata["duration_seconds"] = round(info.length, 3)
        metadata["sample_rate"] = info.sample_rate
        metadata["channels"] = info.channels
        metadata["bit_depth"] = info.bits_per_sample
    elif suffix == ".mp3":
        audio = MP3(file_path)
        info = audio.info
        metadata["duration_seconds"] = round(info.length, 3)
        metadata["sample_rate"] = info.sample_rate
        metadata["channels"] = info.channels
        metadata["bitrate_kbps"] = int(info.bitrate / 1000)

    return metadata


def analyze_with_gemini(file_path: Path, client: genai.Client) -> dict:
    """Send audio to Gemini for content understanding with retry."""
    max_retries = 3
    base_delay = 2

    for attempt in range(max_retries):
        try:
            audio_file = client.files.upload(file=file_path)
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[ANALYSIS_PROMPT, audio_file],
            )

            # Parse JSON response
            text = response.text.strip()
            # Handle markdown code blocks if present
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            result = json.loads(text)

            # Validate and clean category
            category = result.get("category", "other").lower().strip()
            if category not in VALID_CATEGORIES:
                category = "other"

            # Clean tags - ensure they're strings and join with pipe
            tags = result.get("tags", [])
            if isinstance(tags, list):
                tags = "|".join(str(t).strip() for t in tags if t)
            else:
                tags = ""

            return {
                "ai_description": str(result.get("description", "")).strip(),
                "ai_tags": tags,
                "ai_category": category,
            }
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse AI response for {file_path}: {e}", file=sys.stderr)
            if attempt < max_retries - 1:
                time.sleep(base_delay * (2**attempt))
                continue
            return {"ai_description": "", "ai_tags": "", "ai_category": "other"}
        except Exception as e:
            print(f"Warning: AI analysis failed for {file_path}: {e}", file=sys.stderr)
            if attempt < max_retries - 1:
                time.sleep(base_delay * (2**attempt))
                continue
            return {"ai_description": "", "ai_tags": "", "ai_category": "other"}

    return {"ai_description": "", "ai_tags": "", "ai_category": "other"}


def process_file(file_path: Path, client: genai.Client) -> dict | None:
    """Process a single audio file and return catalog entry."""
    try:
        metadata = extract_metadata(file_path)
        ai_analysis = analyze_with_gemini(file_path, client)

        return {
            **metadata,
            **ai_analysis,
            "processed_at": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        print(f"Error processing {file_path}: {e}", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Analyze audio files and output CSV rows to stdout"
    )
    parser.add_argument(
        "--header",
        action="store_true",
        help="Print CSV header row first",
    )
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="Audio files to process",
    )
    args = parser.parse_args()

    # CSV writer to stdout
    writer = csv.DictWriter(sys.stdout, fieldnames=CSV_COLUMNS, extrasaction="ignore")

    if args.header:
        writer.writeheader()

    if not args.files:
        sys.exit(0)

    # Check for API key (only needed when processing files)
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set", file=sys.stderr)
        sys.exit(1)

    # Configure Gemini client
    client = genai.Client(api_key=api_key)

    # Process each file
    errors = 0
    for file_path in args.files:
        if not file_path.exists():
            print(f"Error: File not found: {file_path}", file=sys.stderr)
            errors += 1
            continue

        if file_path.suffix.lower() not in (".wav", ".mp3"):
            print(f"Warning: Skipping unsupported format: {file_path}", file=sys.stderr)
            continue

        entry = process_file(file_path, client)
        if entry:
            writer.writerow(entry)
            sys.stdout.flush()
        else:
            errors += 1

    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
