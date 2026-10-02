#!/usr/bin/env bash
cd "$(dirname "$0")"
[ -x .venv/bin/python ] || { echo "Rode primeiro ./instalar.sh"; exit 1; }
exec .venv/bin/python servidor.py "$@"
