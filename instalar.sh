#!/usr/bin/env bash
# BCast Language Teacher — instalação (Linux / macOS)
set -e
cd "$(dirname "$0")"
PY=$(command -v python3 || command -v python)
[ -z "$PY" ] && { echo "Instale o Python 3.10+ primeiro."; exit 1; }
[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/python -m pip install --upgrade pip -q
.venv/bin/python -m pip install -r requirements.txt -q
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "Placa NVIDIA encontrada: instalando suporte a GPU pro Whisper..."
  .venv/bin/python -m pip install -r requirements-gpu.txt -q || true
fi
if curl -s -m 3 http://127.0.0.1:11434/api/version >/dev/null; then echo "Ollama encontrado."
else echo "ATENÇÃO: Ollama não respondeu. Instale em https://ollama.com"; fi
echo
echo "Baixe o 'cérebro' (uma vez só):"
echo "   ollama pull qwen3:4b     # placa de 6 a 8 GB, ou só CPU (Modo leve)"
echo "   ollama pull qwen3:8b     # placa de 12 GB+ ou Mac M-series com 16 GB+"
echo "O programa escolhe sozinho o melhor modelo pra sua máquina."
echo
echo "Pronto! Para usar:  ./iniciar.sh"
