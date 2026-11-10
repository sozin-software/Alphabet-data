#!/usr/bin/env bash
# Downloads the free dictionary sources into RAW_DIR (default: ../alphabet-data/raw).
set -euo pipefail
RAW="${1:-../alphabet-data/raw}"
mkdir -p "$RAW" && cd "$RAW"
curl -fL -o wiktextract-en.jsonl.gz https://kaikki.org/dictionary/English/kaikki.org-dictionary-English.jsonl.gz
gunzip -kf wiktextract-en.jsonl.gz
curl -fL -o oewn-2025.xml.gz https://en-word.net/downloads/english-wordnet-2025.xml.gz
curl -fL -o moby-3202.zip https://www.gutenberg.org/files/3202/files.zip
unzip -oq moby-3202.zip -d moby
