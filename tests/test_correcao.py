"""
Bateria da correção: frases COM erro precisam ser corrigidas e frases CERTAS não podem
ganhar "correção" inventada. Precisa do Ollama ligado com um modelo baixado.

    python tests/test_correcao.py          (modo normal)
    python tests/test_correcao.py --leve   (modelo leve / CPU)
    python tests/test_correcao.py --es     (espanhol; dá pra juntar com --leve)
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import servidor as S

CASOS = [  # (frase, tem_erro)
    ("It is a RPG like Ragnarok.", True),
    ("The hardest part was make the AI follow the rules.", True),
    ("I have 30 years and I live in Santos.", True),
    ("People is very happy with the app.", True),
    ("Yesterday I go to the beach with my kids.", True),
    ("It depends of the project.", True),
    ("Is very cool to work with games.", True),
    ("I am agree with you.", True),
    ("My day was good, I worked a lot on a game.", False),
    ("I like playing games like Ragnarok.", False),
    ("Yeah, I think it's pretty cool.", False),
    ("I've been working on this project for two months.", False),
    ("What did you think about my game?", False),
    ("um I went to the um supermarket", False),
]

CASOS_ES = [  # portunhol clássico
    ("Yo gusto mucho de fútbol.", True),
    ("Me gusta tomar el leche por la mañana.", True),
    ("La viaje fue muy largo.", True),
    ("Ayer yo voy a la playa con mis hijos.", True),
    ("Estoy con mucha hambre.", True),
    ("Trabajé muy esta semana.", True),
    ("Voy a tirar una foto de la playa.", True),
    ("Muito obrigado por la ayuda.", True),
    ("Ayer fui al cine con mi esposa.", False),
    ("Me gusta mucho la comida mexicana.", False),
    ("Tengo dos hijos y vivo cerca de la playa.", False),
    ("Trabajo con inteligencia artificial desde hace tres años.", False),
    ("Que bueno, me encanta la musica latina.", False),      # sem acentos (Whisper) não é erro
    ("¿Y tú qué haces los fines de semana?", False),
]

def test_regra_a_an():
    assert S.regra_a_an("It is a RPG") == "It is an RPG"
    assert S.regra_a_an("I had a idea") == "I had an idea"
    assert S.regra_a_an("a university") == "a university"
    assert S.regra_a_an("a USB cable") == "a USB cable"

if __name__ == "__main__":
    test_regra_a_an()
    leve, idioma = "--leve" in sys.argv, ("es" if "--es" in sys.argv else "en")
    casos, contexto = (CASOS_ES, "¿Qué tal tu día?") if idioma == "es" else (CASOS, "How was your day?")
    print("Modelo:", S.escolher(leve), "| idioma:", idioma)
    acertos, falsos, t0 = 0, 0, time.time()
    for frase, erro in casos:
        c = S.corrigir(frase, contexto, leve, idioma)
        ok = (c is not None) == erro
        acertos += ok; falsos += (c is not None and not erro)
        print(("OK  " if ok else "XX  ") + frase + (f"  ->  {c['better']}" if c else "  (sem correção)"))
    print(f"\n{acertos}/{len(casos)} certos · {falsos} correções inventadas · {(time.time() - t0) / len(casos):.1f}s por frase")
    sys.exit(0 if falsos == 0 else 1)
