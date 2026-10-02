"""
BCast Language Teacher — converse em inglês ou espanhol por voz, 100% no seu PC.

  ouvir  : faster-whisper (GPU se tiver, senão CPU)
  pensar : Ollama (qualquer modelo instalado; qwen3:8b recomendado) com resposta em JSON
  falar  : Kokoro TTS (onnx, roda na CPU)

Configuração por variáveis de ambiente (todas opcionais):
  SAM_DADOS          pasta dos modelos de voz e cache do Whisper (padrão: ./dados)
  SAM_MODELO         modelo do Ollama no modo normal (padrão: o 1º disponível de uma lista boa)
  SAM_MODELO_LEVE    modelo do Ollama no "modo leve" (CPU)
  SAM_WHISPER        auto | cpu   (padrão auto: tenta GPU, cai pra CPU)
  SAM_WHISPER_MODELO tamanho do Whisper na GPU: medium (padrão) | small | large-v3 ...
  SAM_VRAM_MB        finge que a placa tem X MB (testar a escolha automática)
  SAM_CTX            contexto do modelo em tokens (padrão 8192; 4096 economiza VRAM)
  SAM_PORTA          porta da página (padrão 8770)
  OLLAMA_HOST_URL    onde está o Ollama (padrão http://127.0.0.1:11434)
  SAM_DEBUG=1        grava o log das conversas e libera a rota /log (testes)
"""
import io, json, os, re, sys, tempfile, threading, time, urllib.error, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

AQUI = Path(__file__).resolve().parent
DADOS = Path(os.environ.get("SAM_DADOS", AQUI / "dados"))
MODELOS_VOZ = DADOS / "modelos"
SESSOES = AQUI / "sessoes"
for p in (MODELOS_VOZ, SESSOES): p.mkdir(parents=True, exist_ok=True)
OLLAMA = os.environ.get("OLLAMA_HOST_URL", "http://127.0.0.1:11434").rstrip("/")
PORTA = int(os.environ.get("SAM_PORTA", "8770"))
DEBUG = os.environ.get("SAM_DEBUG") == "1"
CTX = int(os.environ.get("SAM_CTX", "8192"))                    # memória de conversa do modelo (menos = menos VRAM)
os.environ.setdefault("HF_HOME", str(DADOS / "hf"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Preferência de modelos (o 1º que estiver instalado no Ollama ganha)
def vram_total_mb():
    """Memória da placa NVIDIA (None = sem NVIDIA / sem nvidia-smi)."""
    try:
        import subprocess
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout
        return int(out.strip().splitlines()[0])
    except Exception:
        return None
VRAM = int(os.environ["SAM_VRAM_MB"]) if os.environ.get("SAM_VRAM_MB") else vram_total_mb()   # SAM_VRAM_MB simula outra placa
PLACA_GRANDE = bool(VRAM and VRAM >= 11000)      # 12 GB+: qwen3:8b + Whisper medium (~9,6 GB medidos)
# Medido (tests/bench_vram.py): qwen3:8b = 9,6 GB no total; qwen3:4b + Whisper small = 6,0 GB (cabe em 8 GB com folga).
# gemma3:4b ficou de fora: quebra o JSON (fecha a string com aspas tortas).
_GRANDES, _PEQUENOS = ["qwen3:8b", "qwen2.5:7b", "llama3.1:8b"], ["qwen3:4b", "qwen2.5:3b", "llama3.2:3b", "llama3.2:latest"]
PREF_NORMAL = [os.environ.get("SAM_MODELO")] + (_GRANDES + _PEQUENOS if PLACA_GRANDE else _PEQUENOS + _GRANDES)
PREF_LEVE = [os.environ.get("SAM_MODELO_LEVE"), "qwen3:4b", "qwen3:1.7b", "llama3.2:3b", "llama3.2:latest", "qwen2.5:3b"]

# DLLs do CUDA instaladas pelo pip (nvidia-cublas / nvidia-cudnn) — o ctranslate2 precisa achar no Windows
if os.name == "nt":
    for sub in ("cublas", "cudnn", "cuda_runtime"):
        d = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia" / sub / "bin"
        if d.is_dir():
            os.add_dll_directory(str(d)); os.environ["PATH"] = str(d) + os.pathsep + os.environ["PATH"]

# ------------------------------------------------------------------ ouvir
WHISPER = {"m": None, "onde": None}
TRAVA_W = threading.Lock()
def carregar_whisper():
    from faster_whisper import WhisperModel
    if os.environ.get("SAM_WHISPER", "auto") != "cpu":
        try:
            import numpy as np
            nome = os.environ.get("SAM_WHISPER_MODELO") or ("medium" if PLACA_GRANDE else "small")   # small: ~0,8 GB, medium: ~2,1 GB
            m = WhisperModel(nome, device="cuda", compute_type="float16")
            list(m.transcribe(np.zeros(16000, dtype=np.float32), language="en")[0])   # testa o CUDA de verdade
            WHISPER.update(m=m, onde=f"GPU ({nome})")
        except Exception as e:
            print("  Whisper na GPU não rolou (%s) -> CPU" % str(e)[:100])
    if not WHISPER["m"]:
        WHISPER.update(m=WhisperModel("small", device="cpu", compute_type="int8"), onde="CPU (small)")
    print("  Ouvido pronto: Whisper " + WHISPER["onde"])

def transcrever(audio: bytes, idioma="en") -> str:
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
        f.write(audio); nome = f.name
    try:
        with TRAVA_W:
            I = IDIOMAS.get(idioma, IDIOMAS["en"])
            segs, _ = WHISPER["m"].transcribe(nome, language=I["whisper"], beam_size=5, vad_filter=True, initial_prompt=I["dica_whisper"])
            return " ".join(s.text.strip() for s in segs).strip()
    finally:
        os.unlink(nome)

# ------------------------------------------------------------------ falar
KOKORO = {"k": None}
TRAVA_K = threading.Lock()
URL_KOKORO = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
def baixar(nome):
    destino = MODELOS_VOZ / nome
    if destino.exists() and destino.stat().st_size > 1_000_000: return destino
    print(f"  Baixando {nome} (só na 1ª vez)...")
    tmp = destino.with_suffix(".parcial")
    with urllib.request.urlopen(URL_KOKORO + nome, timeout=60) as r, open(tmp, "wb") as f:
        total, feito = int(r.headers.get("Content-Length", 0)), 0
        while bloco := r.read(1 << 20):
            f.write(bloco); feito += len(bloco)
            if total: print(f"\r    {feito * 100 // total}%", end="", flush=True)
    print(); tmp.replace(destino); return destino
def carregar_kokoro():
    from kokoro_onnx import Kokoro
    KOKORO["k"] = Kokoro(str(baixar("kokoro-v1.0.onnx")), str(baixar("voices-v1.0.bin")))
    print("  Voz pronta: Kokoro")

def falar(texto: str, voz: str, vel: float) -> bytes:
    import soundfile as sf
    with TRAVA_K:
        lang = {"a": "en-us", "b": "en-gb", "e": "es"}.get(voz[:1], "en-us")     # 1ª letra da voz = sotaque
        amostras, sr = KOKORO["k"].create(texto, voice=voz, speed=vel, lang=lang)
    buf = io.BytesIO(); sf.write(buf, amostras, sr, format="WAV"); return buf.getvalue()

# ------------------------------------------------------------------ pensar
def http_json(url, corpo=None, timeout=180):
    dados = json.dumps(corpo).encode() if corpo is not None else None
    req = urllib.request.Request(url, data=dados, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

def instalados():
    try: return [m["name"] for m in http_json(OLLAMA + "/api/tags", timeout=3)["models"]]
    except Exception: return []
def escolher(leve):
    tem = instalados()
    for m in (PREF_LEVE if leve else PREF_NORMAL):
        if m and m in tem: return m
    return tem[0] if tem else None

def ollama(messages, schema, leve=False, temperatura=0.7, max_tokens=600):
    modelo = escolher(leve)
    if not modelo: raise RuntimeError("Nenhum modelo no Ollama. Rode:  ollama pull qwen3:4b")
    corpo = {"model": modelo, "messages": messages, "stream": False, "format": schema, "think": False,
             "keep_alive": "30m", "options": {"temperature": temperatura, "num_ctx": CTX, "num_predict": max_tokens}}
    try:
        r = http_json(OLLAMA + "/api/chat", corpo)
    except urllib.error.HTTPError as e:                          # modelo sem "think": tenta sem o campo
        if "think" not in e.read().decode("utf-8", "replace"): raise
        corpo.pop("think"); r = http_json(OLLAMA + "/api/chat", corpo)
    try:
        return json.loads(r["message"]["content"])
    except json.JSONDecodeError:                                  # JSON cortado/torto (modelos pequenos): tenta 1x de novo, mais frio
        corpo["options"]["temperature"] = min(temperatura, 0.3); corpo["options"]["num_predict"] = max_tokens + 300
        r = http_json(OLLAMA + "/api/chat", corpo)
        return json.loads(r["message"]["content"])

# ======================================================== idiomas
# Tudo que muda entre inglês e espanhol mora aqui. As instruções pro modelo ficam em inglês
# (os modelos seguem melhor), mas a fala do tutor sai sempre no idioma escolhido.
IDIOMAS = {
    "en": {
        "lingua": "English", "whisper": "en", "tutor": "Sam",
        "dica_whisper": "Casual conversation practice. The speaker is Brazilian.",
        "persona": "You are Sam, 31, from Seattle, living in Lisbon. You love indie games, hiking, cooking ramen at home and bad sci-fi movies; you work as a UX designer.",
        "lugar": "in the US",
        "erros": ("verb tense/form ('was make' -> 'was making', 'I go yesterday' -> 'I went yesterday'), agreement ('people is' -> 'people are'), "
                  "articles ('a RPG' -> 'an RPG', 'I am engineer' -> 'I am an engineer'), missing subject ('Is very cool' -> 'It is very cool'), "
                  "Portuguese-style phrases ('I have 30 years' -> 'I am 30', 'make a question' -> 'ask a question', 'I am agree' -> 'I agree', 'explain me' -> 'explain to me'), "
                  "false friends ('actually' used as 'atualmente' -> 'currently'), wrong prepositions ('depends of' -> 'depends on')"),
        "nao_erros": "contractions, filler words (um, uh, like as filler), the word 'like' for comparison, informal but correct English, British vs American spelling",
        "livres": {"am", "is", "are", "was", "were", "been", "being", "have", "has", "had", "does", "did", "the", "an", "to", "on", "of", "in", "at",
                   "for", "with", "it", "its", "this", "that", "there", "their", "they", "will", "would", "could", "about", "from", "currently",
                   "agree", "went", "came", "made", "took", "bought", "thought", "said", "told", "asked"},
        "some_pode": {"years", "year", "old", "have", "has", "had", "does", "will", "actually", "make", "makes", "made", "very", "really", "agree"},
    },
    "es": {
        "lingua": "Spanish", "whisper": "es", "tutor": "Alex",
        "dica_whisper": "Conversación casual para practicar español. El hablante es brasileño.",
        "persona": "You are Alex, 29, from Guadalajara (Mexico), living in Madrid. You love salsa dancing, street photography, cooking tacos, football and old telenovelas; you work as a journalist. Use neutral Latin American Spanish (ustedes, not vosotros).",
        "lugar": "in Mexico City or Madrid",
        "erros": ("gender of nouns that differ from Portuguese ('el leche' -> 'la leche', 'la viaje' -> 'el viaje', 'la árbol' -> 'el árbol', 'el sangre' -> 'la sangre', 'la color' -> 'el color', 'el nariz' -> 'la nariz'), "
                  "'gustar' structure ('yo gusto de fútbol' -> 'me gusta el fútbol'), muy vs mucho ('trabajé muy' -> 'trabajé mucho', 'mucho bueno' -> 'muy bueno'), "
                  "contractions ('a el' -> 'al', 'de el' -> 'del'), neuter 'lo' ('el mejor de todo' -> 'lo mejor de todo'), "
                  "verb forms ('ayer yo voy' -> 'ayer fui', 'yo sabo' -> 'yo sé'), pronoun position ('llamo-me' -> 'me llamo'), "
                  "Portuguese-style phrases ('estoy con hambre' -> 'tengo hambre', 'tirar una foto' -> 'sacar una foto'), "
                  "false friends ('me acordé a las siete' meaning woke up -> 'me desperté', 'exquisito' meaning strange -> 'raro', 'oficina' meaning workshop -> 'taller', "
                  "'embarazada' meaning embarrassed -> 'avergonzada', 'rato' meaning mouse -> 'ratón', 'borracha' meaning eraser -> 'goma'), "
                  "Portuguese words mixed in ('muito' -> 'mucho/muy', 'você' -> 'tú/usted', 'então' -> 'entonces', 'obrigado' -> 'gracias', 'ontem' -> 'ayer', 'fazer' -> 'hacer')"),
        "nao_erros": "missing accents or ñ (speech recognition), tú vs usted vs vos, ustedes vs vosotros, regional vocabulary (coche/carro/auto), informal but correct Spanish",
        "livres": {"está", "están", "estoy", "estás", "somos", "fue", "fui", "fuimos", "eran", "unos", "unas", "para", "por", "muy", "mucho", "mucha",
                   "muchos", "muchas", "tengo", "tiene", "tienes", "gusta", "gustan", "encanta", "sacar", "saqué", "tomar", "tomé", "coger",
                   "despertar", "desperté", "raro", "extraño", "taller", "ratón", "goma", "avergonzado", "avergonzada", "entonces", "gracias",
                   "ayer", "hacer", "hice", "usted", "este", "esta", "esto", "también", "pero", "cuando", "donde", "porque", "sobre", "hasta"},
        "some_pode": {"estoy", "estaba", "gusto", "gustar", "muy", "mucho", "tirar", "tiré", "pegar", "acordar", "acordé", "exquisito", "esquisito",
                      "oficina", "rato", "borracha", "embarazada", "embarazado", "muito", "você", "então", "obrigado", "obrigada", "ontem",
                      "fazer", "depois", "coisa", "mais", "agora", "hoje", "trabalho", "também"},
    },
}
def L(d): return IDIOMAS.get((d or {}).get("idioma") if isinstance(d, dict) else d, IDIOMAS["en"])

CENAS = {
    "livre": "a friendly casual chat: day, family, hobbies, work, games, music, food. Be curious and warm.",
    "entrevista": "a job interview in {lingua} for a position in the learner's field (use what you know about them; if unknown, ask). You are the interviewer: one question at a time (experience, projects, strengths, teamwork, salary expectations), react like a real recruiter.",
    "reuniao": "a daily stand-up / work meeting with a remote international team. You are the team lead: ask what they did yesterday, blockers, plans; sometimes ask for clarification.",
    "restaurante": "ordering food at a restaurant/coffee shop {lugar}. You are the waiter: greet, take the order, suggest dishes, handle questions and the bill.",
    "viagem": "a travel situation {lugar}: airport check-in, immigration, hotel reception, asking for directions. You play the staff member; change the situation naturally.",
}
NIVEIS = {
    "basico": "The learner is a BEGINNER (A2). Use short, simple sentences, very common words. Max 2 short sentences.",
    "intermediario": "The learner is INTERMEDIATE (B1). Use natural everyday {lingua}, common expressions, max 3 sentences.",
    "avancado": "The learner is ADVANCED (B2-C1). Use fully natural {lingua}, idioms and richer vocabulary, max 3-4 sentences.",
}
def schema_chat(com_traducao):
    props = {"reply": {"type": "string"},
             "new_words": {"type": "array", "items": {"type": "object", "properties": {
                 "en": {"type": "string"}, "pt": {"type": "string"}}, "required": ["en", "pt"]}}}
    if com_traducao: props["reply_pt"] = {"type": "string"}
    return {"type": "object", "properties": props, "required": list(props)}

def quem_e(perfil):
    nome = (perfil or {}).get("nome", "").strip()[:40] or "the learner"
    sobre = (perfil or {}).get("sobre", "").strip()[:400]
    return nome, (f" What you know about {nome}: {sobre}." if sobre else "")

def sistema(d, com_traducao):
    I = L(d); lingua = I["lingua"]
    nome, sobre = quem_e(d.get("perfil"))
    cena = CENAS.get(d.get("cena"), CENAS["livre"]).format(lingua=lingua, lugar=I["lugar"])
    nivel = NIVEIS.get(d.get("nivel"), NIVEIS["intermediario"]).format(lingua=lingua)
    return (f"{I['persona']} You are talking with {nome}, a Brazilian who is practicing {lingua}.{sobre}\n"
            f"ALWAYS speak {lingua} only.\nScenario: {cena}\n{nivel}\n"
            "HOW TO TALK (very important):\n"
            "- This is ONE continuous conversation. Always read the whole history and build on it. Remember names, facts and details told earlier and bring them back later.\n"
            "- React to the SPECIFIC detail in the last message first (quote or paraphrase it), then add something of your own: an opinion, a short personal story, a joke or a related experience.\n"
            "- Stay on the current topic for several turns, going DEEPER (why, how, what happened next, how did you feel). Only change topic if the learner does, or if it is clearly finished; when you change, connect it to something said before.\n"
            "- Ask at most ONE question per reply, and it must follow from what was just said. Sometimes don't ask anything, just comment.\n"
            "- Never repeat a question you already asked. Never answer with only praise.\n"
            f"- Do NOT correct the learner's {lingua} here (another system does that). If they speak Portuguese, answer in {lingua} and invite them to try in {lingua}.\n"
            "- 'reply' is read aloud: no emojis, no markdown, no lists.\n"
            + ("- 'reply_pt': faithful Brazilian Portuguese translation of 'reply'.\n" if com_traducao else "")
            + f"- 'new_words': 0 to 2 useful {lingua} words/expressions YOU used in 'reply' that may be new for the learner's level ('en' = the {lingua} word), with Portuguese meaning ('pt').")

# ----------------------------------------------------- correção
SCHEMA_CORR = {"type": "object", "properties": {"corrected": {"type": "string"}, "why_pt": {"type": "string"}},
               "required": ["corrected", "why_pt"]}
def _norm(t):
    t = t.lower()
    for a, b in (("’", "'"), ("i'm", "i am"), ("it's", "it is"), ("don't", "do not"), ("didn't", "did not"), ("that's", "that is"),
                 ("i've", "i have"), ("can't", "cannot"), ("won't", "will not"), ("isn't", "is not")):
        t = t.replace(a, b)
    for a, b in (("á", "a"), ("é", "e"), ("í", "i"), ("ó", "o"), ("ú", "u"), ("ñ", "n"), ("ü", "u")):   # acento faltando NÃO é erro (é o Whisper)
        t = t.replace(a, b)
    return "".join(ch for ch in t if ch.isalnum())

SOM_VOGAL_SIGLA = set("AEFHILMNORSX")                             # letras que, soletradas, começam com som de vogal
def regra_a_an(frase):
    """'a RPG' -> 'an RPG', 'a idea' -> 'an idea' (modelo pequeno costuma deixar passar). Só inglês."""
    def troca(m):
        art, palavra = m.group(1), m.group(2)
        sigla = palavra.isupper() and len(palavra) > 1
        vogal = palavra[0] in SOM_VOGAL_SIGLA if sigla else (palavra[0].lower() in "aeio" and not palavra.lower().startswith(("one", "once", "eu", "uni", "use", "usu")))
        return (("An " if art == "A" else "an ") + palavra) if vogal else m.group(0)
    return re.sub(r"(?<![A-Za-z])([Aa]) ([A-Za-z][A-Za-z0-9-]*)", troca, frase)

def _tokens(t):
    return re.findall(r"[^\W_]+", t.lower().replace("’", "'"))
def reescrita_suspeita(frase, certo, idioma="en"):
    """Modelo pequeno às vezes REESCREVE a frase (troca 'game' por 'RPG', inventa 'yesterday').
    Correção de verdade mexe pouco: não some com palavra de conteúdo nem inventa palavra nova."""
    I = L(idioma)
    a, b = _tokens(frase), _tokens(certo)
    parecido = lambda w, lista: any(_norm(w)[:3] == _norm(x)[:3] for x in lista)
    novas = [w for w in b if w not in a and len(w) > 3 and w not in I["livres"] and not parecido(w, a)]
    sumiram = [w for w in a if w not in b and len(w) > 3 and w not in I["some_pode"] and not parecido(w, b)]
    return bool(novas or sumiram)

def parece_portugues(t, idioma="en"):
    """Explicação que não está em português = modelo se perdeu (acontece com os pequenos)."""
    t = " " + t.lower() + " "
    so_pt = (" não ", " em ", " é ", " uma ", " um ", " você", " faltou", " artigo", " sujeito", " correto", " depois", " do ", " da ",
             " usa-se", " certo ", " palavra", " forma ", " porque ", " mudou", " mudei", " deveria", " seria ", " verbo ")
    if idioma == "es":                                            # espanhol também tem acento, 'de', 'que'... só vale o que é SÓ português
        return any(c in t for c in "ãõç") or sum(w in t for w in so_pt) >= 2
    return any(c in t for c in "ãõçéêáíóú") or any(w in t for w in so_pt + (" de ", " que ", " para ", " com ", " o ", " os ", " antes ", " deve "))

SCHEMA_SIMNAO = {"type": "object", "properties": {"is_wrong": {"type": "boolean"}}, "required": ["is_wrong"]}
def tem_erro(frase, leve=False, idioma="en"):
    I = L(idioma)
    msgs = [{"role": "system", "content": f"You are a {I['lingua']} grammar checker. Answer is_wrong=true ONLY if the sentence has a clear grammar or word-choice mistake that a native speaker would never make. Ignore punctuation, capitalization, accents, filler words and informal style."},
            {"role": "user", "content": f"Sentence: {frase}"}]
    return bool(ollama(msgs, SCHEMA_SIMNAO, leve, 0.0, 20).get("is_wrong"))

def corrigir(frase, contexto, leve=False, idioma="en"):
    """A IA só REESCREVE a frase do jeito certo; quem decide se havia erro é o CÓDIGO (comparando).
    Isso derruba as 'correções' inventadas que os modelos pequenos adoram fazer."""
    I = L(idioma)
    msgs = [{"role": "system", "content":
             f"You are a strict but fair {I['lingua']} teacher for a Brazilian learner. Rewrite the learner's sentence fixing ONLY real mistakes, "
             "keeping their words and meaning; change as little as possible. If the sentence is already correct (even if informal), return it EXACTLY unchanged.\n"
             f"Real mistakes include: {I['erros']}.\n"
             f"NOT mistakes (never change): punctuation, capitalization, {I['nao_erros']}. The text comes from speech recognition.\n"
             "'why_pt': if you changed something, explain each change in short BRAZILIAN PORTUGUESE (max 2 sentences); if unchanged, empty string."},
            {"role": "user", "content": f"(Context, the other person said: {contexto})\nLearner's sentence: {frase}"}]
    r = ollama(msgs, SCHEMA_CORR, leve, 0.0, 250)
    certo = (r.get("corrected") or "").strip()
    por_que = (r.get("why_pt") or "").strip()
    mudou = certo and _norm(certo) != _norm(frase) and not reescrita_suspeita(frase, certo, idioma)
    if mudou and not parece_portugues(por_que, idioma):
        # explicação fora do português = sinal de que o modelo se perdeu; só aceita se ele CONFIRMAR o erro num sim/não
        if tem_erro(frase, leve, idioma):
            por_que = "Compare as duas frases: a de baixo é o jeito correto de dizer."   # traduzir com modelo pequeno sai pior
        else:
            mudou = False
    if idioma == "en":
        if not mudou:
            certo2 = regra_a_an(frase)
            if certo2 != frase:
                return {"original": frase, "better": certo2, "why_pt": "Antes de palavra (ou sigla) com som de vogal usa-se 'an', não 'a' — ex.: 'an RPG', 'an apple'."}
            return None
        certo = regra_a_an(certo)
    if not mudou:
        return None
    return {"original": frase, "better": certo, "why_pt": por_que}

EMOJI = re.compile("[🀀-🫿☀-➿️‍]")
def sem_emoji(t): return re.sub(r"\s{2,}", " ", EMOJI.sub("", t)).strip()

# ----------------------------------------------------- conversa
def conversar(d):
    leve, idioma = bool(d.get("leve")), d.get("idioma", "en")
    historico = d.get("historico", [])
    hist = historico[-24:]
    if DEBUG:
        with open(SESSOES / "_conversa.log", "a", encoding="utf-8") as f:
            f.write(time.strftime("%H:%M:%S") + f" [{idioma}] recebi {len(historico)} falas | ultima: " + (hist[-1]["texto"][:80] if hist else "-") + "\n")
    nome, _ = quem_e(d.get("perfil"))
    msgs = [{"role": "system", "content": sistema(d, not leve)}]
    for h in hist:
        msgs.append({"role": "user" if h["quem"] == "eu" else "assistant", "content": h["texto"]})
    if not hist:                                                  # o tutor começa a conversa
        msgs.append({"role": "user", "content": f"(Start the conversation now, in {L(d)['lingua']}: greet {nome} warmly and open the scenario with one question.)"})
    corr, t = {}, None
    if hist and hist[-1]["quem"] == "eu":
        anterior = next((h["texto"] for h in reversed(hist[:-1]) if h["quem"] != "eu"), "")
        def _c():
            try: corr["c"] = corrigir(hist[-1]["texto"], anterior, leve, idioma)
            except Exception as e: print("  correção falhou:", e)
        t = threading.Thread(target=_c)
        if not leve: t.start()                                    # GPU: correção em paralelo com a resposta
    r = ollama(msgs, schema_chat(not leve), leve, 0.8)
    r["reply"] = sem_emoji(r.get("reply", ""))                       # vai ser lido em voz alta
    if t:
        if leve: t.start()                                        # CPU: um de cada vez (senão disputam os núcleos)
        t.join()
    r["correction"] = corr.get("c")
    return r

SCHEMA_TRAD = {"type": "object", "properties": {"pt": {"type": "string"}}, "required": ["pt"]}
def traduzir(d):
    msgs = [{"role": "system", "content": f"Translate the {L(d)['lingua']} text into natural Brazilian Portuguese. Output only the translation."},
            {"role": "user", "content": d.get("texto", "")}]
    return ollama(msgs, SCHEMA_TRAD, bool(d.get("leve")), 0.2, 300)

SCHEMA_AJUDA = {"type": "object", "properties": {"options": {"type": "array", "items": {"type": "object", "properties": {
    "en": {"type": "string"}, "pt": {"type": "string"}}, "required": ["en", "pt"]}}}, "required": ["options"]}
def ajudar(d):
    I = L(d); nome, sobre = quem_e(d.get("perfil"))
    hist = d.get("historico", [])[-8:]
    conversa = "\n".join((f"{nome}: " if h["quem"] == "eu" else f"{I['tutor']}: ") + h["texto"] for h in hist)
    msgs = [{"role": "system", "content": f"You help a Brazilian {I['lingua']} learner who is stuck in a conversation." + sobre + " " + NIVEIS.get(d.get("nivel"), NIVEIS["intermediario"]).format(lingua=I["lingua"])},
            {"role": "user", "content": f"Conversation so far:\n{conversa}\n\nGive 3 different short things {nome} could say next IN {I['lingua'].upper()} (simple, natural, varied: one answer, one with a detail, one asking back). 'en' = the sentence in {I['lingua']}, 'pt' = Portuguese translation."}]
    return ollama(msgs, SCHEMA_AJUDA, bool(d.get("leve")), 0.9, 300)

SCHEMA_RESUMO = {"type": "object", "properties": {
    "nota": {"type": "integer"}, "pontos_fortes": {"type": "string"},
    "erros": {"type": "array", "items": {"type": "object", "properties": {"errado": {"type": "string"}, "certo": {"type": "string"}, "dica": {"type": "string"}}, "required": ["errado", "certo", "dica"]}},
    "proximo_passo": {"type": "string"}}, "required": ["nota", "pontos_fortes", "erros", "proximo_passo"]}
def resumir(d):
    I = L(d); nome, _ = quem_e(d.get("perfil"))
    falas = "\n".join((f"{nome}: " if h["quem"] == "eu" else f"{I['tutor']}: ") + h["texto"] for h in d.get("historico", []))
    corr = json.dumps(d.get("correcoes", []), ensure_ascii=False)
    msgs = [{"role": "system", "content": f"You are a {I['lingua']} teacher writing feedback IN BRAZILIAN PORTUGUESE for a learner after a conversation session. Be encouraging and concrete. "
             "You only see speech-to-text transcripts: NEVER comment on pronunciation, accent or intonation."},
            {"role": "user", "content": f"Session ({d.get('cena')}, level {d.get('nivel')}):\n{falas}\n\nCorrections made during the session: {corr}\n\n"
             "Return: 'nota' fluency 1-10; 'pontos_fortes' (1-2 sentences pt-BR); 'erros' = the most important DISTINCT mistakes (max 5, never repeat the same mistake, only real errors, with a tiny tip in pt-BR each); "
             f"'proximo_passo' = one practical thing to practice IN {I['lingua'].upper()} next time, written in pt-BR (e.g. a structure to use, a topic to talk about)."}]
    r = ollama(msgs, SCHEMA_RESUMO, bool(d.get("leve")), 0.4, 700)
    vistos, unicos = set(), []                                    # o modelo às vezes repete o mesmo erro
    for e in r.get("erros", []):
        k = e.get("certo", "").strip().lower().rstrip(".!?")
        if k and k not in vistos and k != e.get("errado", "").strip().lower().rstrip(".!?"):
            vistos.add(k); unicos.append(e)
    r["erros"] = unicos
    arq = time.strftime("%Y-%m-%d_%H%M") + "_" + re.sub(r"\W", "", str(d.get("idioma", "en"))) + "_" + re.sub(r"\W", "", str(d.get("cena", "livre"))) + ".json"
    salvar = {k: v for k, v in d.items() if k != "perfil"}
    (SESSOES / arq).write_text(json.dumps({"quando": time.strftime("%Y-%m-%d %H:%M"), **salvar, "resumo": r}, ensure_ascii=False, indent=1), encoding="utf-8")
    return r

def historico_sessoes():
    out = []
    for p in sorted(SESSOES.glob("*.json"), reverse=True)[:30]:
        try:
            s = json.loads(p.read_text(encoding="utf-8"))
            out.append({"quando": s.get("quando"), "cena": s.get("cena"), "nivel": s.get("nivel"), "idioma": s.get("idioma", "en"),
                        "falas": sum(1 for h in s.get("historico", []) if h["quem"] == "eu"),
                        "nota": s.get("resumo", {}).get("nota"), "erros": s.get("resumo", {}).get("erros", [])})
        except Exception:
            pass
    return out

# ------------------------------------------------------------------ http
TIPOS = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css",
         ".svg": "image/svg+xml", ".png": "image/png", ".ico": "image/x-icon"}
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def responder(self, cod, corpo, tipo="application/json; charset=utf-8"):
        if isinstance(corpo, (dict, list)): corpo = json.dumps(corpo, ensure_ascii=False).encode()
        self.send_response(cod); self.send_header("Content-Type", tipo); self.send_header("Content-Length", str(len(corpo)))
        self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(corpo)
    def do_GET(self):
        caminho = self.path.split("?")[0]
        if caminho == "/status":
            tem = instalados()
            return self.responder(200, {"ollama": bool(tem) or self._ollama_vivo(), "modelos": tem,
                                        "modelo": escolher(False), "modelo_leve": escolher(True),
                                        "whisper": WHISPER["onde"], "voz": bool(KOKORO["k"]), "vram_mb": VRAM})
        if caminho == "/sessoes": return self.responder(200, historico_sessoes())
        raiz = (AQUI / "web").resolve()
        arq = (raiz / (caminho.strip("/") or "index.html")).resolve()
        if raiz not in arq.parents or not arq.is_file():           # nada fora da pasta web/
            return self.responder(404, {"erro": "não achei"})
        self.responder(200, arq.read_bytes(), TIPOS.get(arq.suffix, "application/octet-stream"))
    def _ollama_vivo(self):
        try: http_json(OLLAMA + "/api/version", timeout=2); return True
        except Exception: return False
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0)); corpo = self.rfile.read(n)
        caminho = self.path.split("?")[0]
        try:
            if caminho == "/log" and DEBUG:
                with open(SESSOES / "_conversa.log", "a", encoding="utf-8") as f: f.write("PAGINA: " + corpo.decode("utf-8", "replace") + "\n")
                return self.responder(200, {"ok": True})
            if caminho == "/ouvir":
                if not WHISPER["m"]: return self.responder(503, {"erro": "o ouvido (Whisper) ainda está carregando"})
                idioma = "es" if "idioma=es" in self.path else "en"
                return self.responder(200, {"texto": transcrever(corpo, idioma)})
            d = json.loads(corpo or b"{}")
            if caminho == "/falar":
                if not KOKORO["k"]: return self.responder(503, {"erro": "a voz ainda está carregando"})
                return self.responder(200, falar(d["texto"], d.get("voz", "af_heart"), float(d.get("vel", 1.0))), "audio/wav")
            rotas = {"/conversar": conversar, "/traduzir": traduzir, "/ajuda": ajudar, "/resumo": resumir}
            if caminho in rotas: return self.responder(200, rotas[caminho](d))
            self.responder(404, {"erro": "rota?"})
        except urllib.error.URLError as e:
            self.responder(502, {"erro": "Ollama fora do ar (" + str(getattr(e, "reason", e)) + "). Ele está aberto?"})
        except Exception as e:
            self.responder(500, {"erro": str(e)[:300]})

if __name__ == "__main__":
    print("BCast Language Teacher — ligando...")
    threading.Thread(target=carregar_whisper, daemon=True).start()
    threading.Thread(target=carregar_kokoro, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", PORTA), H)            # já escutando -> pode abrir o navegador
    print(f"  Abra http://127.0.0.1:{PORTA}/   (feche esta janela pra desligar)")
    m = escolher(False)
    print(f"  Placa: {VRAM} MB" if VRAM else "  Placa: sem NVIDIA (tudo na CPU; ative o Modo leve)")
    print(f"  Cérebro: {m}" if m else "  ATENÇÃO: nenhum modelo no Ollama. Instale o Ollama e rode:  ollama pull qwen3:4b   (12 GB+ de placa: qwen3:8b)")
    if "--sem-navegador" not in sys.argv:
        import webbrowser; webbrowser.open(f"http://127.0.0.1:{PORTA}/")
    srv.serve_forever()
