#!/usr/bin/env bash
# Download QQP (GLUE) and the BERT-large-cased vocab used by this repo.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${ROOT}/task_datasets/data"
QQP_DIR="${DATA_DIR}/QQP"
VOCAB_PATH="${DATA_DIR}/bert-large-cased-vocab.txt"

QQP_URL="${QQP_URL:-https://dl.fbaipublicfiles.com/glue/data/QQP.zip}"
VOCAB_URL="${VOCAB_URL:-https://huggingface.co/bert-large-cased/resolve/main/vocab.txt}"
VOCAB_URL_FALLBACK="https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-vocab.txt"

download() {
  local url="$1"
  local dest="$2"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 3 --retry-delay 2 -o "${dest}" "${url}"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "${dest}" "${url}"
  else
    echo "Need curl or wget to download ${url}" >&2
    exit 1
  fi
}

mkdir -p "${DATA_DIR}"

if [[ -f "${VOCAB_PATH}" ]]; then
  echo "Vocab already present: ${VOCAB_PATH}"
else
  echo "Downloading BERT vocab..."
  if ! download "${VOCAB_URL}" "${VOCAB_PATH}"; then
    echo "Primary vocab URL failed, trying fallback..."
    download "${VOCAB_URL_FALLBACK}" "${VOCAB_PATH}"
  fi
fi

if [[ -f "${QQP_DIR}/train.tsv" && -f "${QQP_DIR}/dev.tsv" && -f "${QQP_DIR}/test.tsv" ]]; then
  echo "QQP already present: ${QQP_DIR}"
else
  echo "Downloading GLUE QQP (~40 MB zip)..."
  tmp_zip="$(mktemp "${DATA_DIR}/QQP.XXXXXX.zip")"
  download "${QQP_URL}" "${tmp_zip}"
  mkdir -p "${DATA_DIR}"
  unzip -o "${tmp_zip}" -d "${DATA_DIR}"
  rm -f "${tmp_zip}"
  if [[ ! -f "${QQP_DIR}/train.tsv" ]]; then
    echo "QQP zip extracted but ${QQP_DIR}/train.tsv was not found." >&2
    echo "Contents of ${DATA_DIR}:" >&2
    ls -la "${DATA_DIR}" >&2
    exit 1
  fi
fi

echo "Data ready:"
ls -lh "${VOCAB_PATH}" "${QQP_DIR}/train.tsv" "${QQP_DIR}/dev.tsv" "${QQP_DIR}/test.tsv"
