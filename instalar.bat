@echo off
title BCast Language Teacher - instalacao
cd /d "%~dp0"
echo.
echo  === BCast Language Teacher: instalacao ===
echo.

where py >nul 2>nul && (set PY=py -3) || (set PY=python)
%PY% --version >nul 2>nul || (echo  [ERRO] Python nao encontrado. Instale o Python 3.10+ em https://www.python.org/downloads/ e marque "Add to PATH". & pause & exit /b 1)

if not exist .venv (
  echo  Criando ambiente Python em .venv ...
  %PY% -m venv .venv || (echo  [ERRO] Nao consegui criar o .venv & pause & exit /b 1)
)
echo  Instalando dependencias (pode levar alguns minutos)...
.venv\Scripts\python -m pip install --upgrade pip -q
.venv\Scripts\python -m pip install -r requirements.txt -q || (echo  [ERRO] Falhou o pip install & pause & exit /b 1)

where nvidia-smi >nul 2>nul && (
  echo  Placa NVIDIA encontrada: instalando suporte a GPU pro Whisper...
  .venv\Scripts\python -m pip install -r requirements-gpu.txt -q
)

echo.
curl -s -m 3 http://127.0.0.1:11434/api/version >nul 2>nul && (
  echo  Ollama encontrado.
) || (
  echo  [ATENCAO] Ollama nao respondeu. Instale em https://ollama.com e abra ele.
)
echo.
echo  Falta baixar o "cerebro" (uma vez so). No terminal, rode UM destes:
echo     ollama pull qwen3:4b      (placa de video de 6 a 8 GB, ou sem placa no Modo leve)
echo     ollama pull qwen3:8b      (placa de video com 12 GB ou mais)
echo  O programa escolhe sozinho o melhor modelo pra sua placa.
echo.
echo  Pronto! Para usar, de dois cliques em iniciar.bat
echo  (os modelos de voz, ~350 MB, baixam sozinhos na primeira vez)
echo.
pause
