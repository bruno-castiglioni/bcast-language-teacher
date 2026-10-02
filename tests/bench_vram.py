"""
Compara modelos pra placa de 8 GB: memória de vídeo (Whisper + LLM juntos), correção e conversa.

    python tests/bench_vram.py qwen3:4b gemma3:4b qwen3:8b
    SAM_WHISPER_MODELO=small python tests/bench_vram.py qwen3:4b

Precisa de placa NVIDIA (nvidia-smi) e do Ollama rodando LOCAL (mesma placa).
"""
import json, os, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import servidor as S
from test_correcao import CASOS, CASOS_ES

def vram_mb():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    return int(out.strip().splitlines()[0])

def descarregar_tudo():
    try:
        for m in S.http_json(S.OLLAMA + "/api/ps", timeout=5).get("models", []):
            S.http_json(S.OLLAMA + "/api/generate", {"model": m["name"], "keep_alive": 0})
    except Exception as e:
        print("  (não consegui descarregar:", e, ")")
    time.sleep(3)

CONVERSA = {
    "en": ["My day was good, I worked a lot on a new design.", "It is an app for a bakery, I love bread.", "Do you remember the name of my cat?"],
    "es": ["Mi día fue bueno, trabajé mucho en un nuevo diseño.", "Es una app para una panadería, me encanta el pan.", "¿Te acuerdas del nombre de mi gato?"],
}
PERFIL = {"nome": "Ana", "sobre": "Sou designer em Recife e tenho um gato chamado Pixel."}

def conversa(idioma):
    h, tempos, falas = [], [], []
    for i in range(len(CONVERSA[idioma]) + 1):
        t = time.time()
        r = S.conversar({"historico": h, "idioma": idioma, "cena": "livre", "nivel": "intermediario", "perfil": PERFIL})
        tempos.append(time.time() - t)
        h.append({"quem": "sam", "texto": r["reply"]}); falas.append(r["reply"])
        if i < len(CONVERSA[idioma]): h.append({"quem": "eu", "texto": CONVERSA[idioma][i]})
    return falas, tempos

def bateria(casos, idioma, contexto):
    ok = inventadas = 0
    for frase, erro in casos:
        c = S.corrigir(frase, contexto, False, idioma)
        ok += (c is not None) == erro; inventadas += (c is not None and not erro)
    return ok, inventadas

if __name__ == "__main__":
    modelos = sys.argv[1:] or ["qwen3:4b", "gemma3:4b"]
    descarregar_tudo()
    base = vram_mb()
    print(f"Placa em repouso: {base} MB")
    S.carregar_whisper()
    com_whisper = vram_mb()
    print(f"Whisper {S.WHISPER['onde']}: +{com_whisper - base} MB")
    resultado = []
    for m in modelos:
        descarregar_tudo()
        S.PREF_NORMAL[:] = [m]
        print(f"\n===== {m}")
        falas_en, t_en = conversa("en")
        pico = vram_mb()
        falas_es, t_es = conversa("es")
        pico = max(pico, vram_mb())
        b_en = bateria(CASOS, "en", "How was your day?")
        b_es = bateria(CASOS_ES, "es", "¿Qué tal tu día?")
        lembrou = ("pixel" in falas_en[-1].lower(), "pixel" in falas_es[-1].lower())
        linha = {"modelo": m, "vram_total_mb": pico, "vram_app_mb": pico - base,
                 "resp_media_s": round(sum(t_en[1:] + t_es[1:]) / len(t_en[1:] + t_es[1:]), 1),
                 "correcao_en": f"{b_en[0]}/14 ({b_en[1]} inventadas)", "correcao_es": f"{b_es[0]}/14 ({b_es[1]} inventadas)",
                 "lembrou_pixel": lembrou}
        resultado.append(linha)
        print(json.dumps(linha, ensure_ascii=False))
        print("  EN:", " || ".join(f[:110] for f in falas_en))
        print("  ES:", " || ".join(f[:110] for f in falas_es))
    print("\nRESUMO")
    for l in resultado: print(json.dumps(l, ensure_ascii=False))
