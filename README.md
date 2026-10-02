<p align="center"><img src="web/icone.svg" width="72" alt=""></p>

<h1 align="center">BCast Language Teacher</h1>

<p align="center"><i>Seus tutores offline: a <b>Sam</b> 🇺🇸 e o <b>Alex</b> 🇪🇸</i></p>

<p align="center"><b>Converse em inglês 🇺🇸 ou espanhol 🇪🇸 por voz, receba correção na hora e explicação em português.<br>
100% no seu PC: grátis, sem assinatura, sem chave de API e sem mandar sua voz pra ninguém.</b></p>

<!-- VÍDEO / GIF DE DEMONSTRAÇÃO: arraste o arquivo aqui na edição do README pelo GitHub -->

---

## O que ela faz

- **Inglês ou espanhol.** Em inglês você conversa com a **Sam** (americana morando em Lisboa); em espanhol, com o **Alex** (mexicano morando em Madri). Troca no seletor *Idioma*, a qualquer momento.
- **Você fala, o tutor responde em voz alta.** Segura o botão (ou a barra de espaço), fala, solta. Ela ouve, pensa e responde em uns 4 segundos.
- **Correção sem enrolação.** Errou? Aparece logo abaixo da sua fala: ~~o que você disse~~ → **o jeito certo**, com uma explicação curta em português e um botão pra ouvir a frase correta.
- **Feita pra quem fala português.** Conhece os erros clássicos de brasileiro:
  - inglês: *"I have 30 years"*, *"people is"*, *"depends of"*, *"actually"* no sentido de "atualmente", *"Is very cool"* sem sujeito, *"a RPG"*...
  - espanhol (o famoso portunhol): *"yo gusto de"*, *"el leche"*, *"la viaje"*, *"trabajé muy"*, *"estoy con hambre"*, *"tirar una foto"*, *"muito obrigado"*...
- **Conversa de verdade, não questionário.** O tutor tem personalidade, lembra do que você contou, opina e aprofunda o assunto em vez de pular de pergunta em pergunta.
- **Cenas:** papo livre · entrevista de emprego · reunião de trabalho · restaurante · viagem (aeroporto, hotel).
- **Níveis:** básico, intermediário e avançado.
- **Travou?** O botão *Me ajuda* sugere 3 coisas que você poderia dizer.
- **Cada fala do tutor** pode ser ouvida de novo, mais devagar, ou traduzida (PT).
- **No fim da conversa:** nota de fluência, os erros que você mais cometeu e o que treinar na próxima vez. Fica tudo no histórico.
- **9 vozes** (6 em inglês, americanas e britânicas; 3 em espanhol) e 4 velocidades.

## Como funciona

```
 você fala ──► Whisper (ouvido) ──► Ollama / LLM local (cérebro) ──► Kokoro (voz) ──► tutor responde
                                         │
                                         └──► correção da sua frase (em paralelo)
```

O truque da correção: modelos pequenos adoram **inventar erro** em frase certa. Aqui a IA só reescreve sua frase do jeito correto, e quem decide se havia erro é o **código**, comparando as duas versões (mais uma regra fixa pra *a/an*). Bateria de testes (`tests/test_correcao.py`), 14 frases por idioma, metade com erro e metade certa:

| Modelo | Inglês | Espanhol | Correções inventadas |
|---|---|---|---|
| `qwen3:8b` | **14/14** | **14/14** | 0 |
| `qwen3:4b` | **14/14** | 12–13/14 | 0 |
| `llama3.2` 3B | 11–12/14 | 6/14 | 0 |

Os 3B pequenos às vezes "consertam" pra pior; as travas preferem não mostrar nada a mostrar uma correção errada.

## Instalação

Você precisa de:

1. **Python 3.10+**: [python.org](https://www.python.org/downloads/) (no Windows, marque *Add to PATH*).
2. **Ollama**: [ollama.com](https://ollama.com). Depois baixe um "cérebro" (uma vez só):

   | Seu PC | Comando | Memória de vídeo usada* | Cada resposta |
   |---|---|---|---|
   | Placa com **12 GB+** | `ollama pull qwen3:8b` | ~8,2 GB | ~3 s |
   | Placa com **6 a 8 GB** (a maioria) | `ollama pull qwen3:4b` | ~4,6 GB | ~3 s |
   | **Sem placa** / notebook | `ollama pull qwen3:4b` e ative o **Modo leve** em *Você* | — (CPU) | ~10–20 s |

   \* Medido numa RTX 5060 Ti com `tests/bench_vram.py`: ouvido (Whisper) + cérebro carregados juntos, sem contar o ~1,4 GB que o Windows já usa. **O programa olha sua placa e escolhe sozinho** o modelo e o tamanho do ouvido (com menos de 11 GB: `qwen3:4b` + Whisper `small`). Mac com Apple Silicon roda pelo Ollama normalmente (16 GB+ de memória: `qwen3:8b`).

   No modo leve a conversa fica um pouco mais simples e algumas correções vêm sem a explicação detalhada (o modelo pequeno explica mal em português, então só mostra o jeito certo).

3. **Este projeto:**

   ```bash
   git clone https://github.com/bruno-castiglioni/bcast-language-teacher.git
   cd bcast-language-teacher
   ```

   - **Windows:** dois cliques em `instalar.bat`, depois em `iniciar.bat`.
   - **Linux / macOS:** `./instalar.sh` e depois `./iniciar.sh`.

O navegador abre sozinho em `http://127.0.0.1:8770`. Na primeira vez, as vozes (~350 MB) e o ouvido (Whisper) são baixados automaticamente. Depois disso, **funciona sem internet**.

> O navegador vai pedir permissão pro microfone. Use Chrome ou Edge.

## Configuração (opcional)

Tudo por variável de ambiente:

| Variável | Pra quê | Padrão |
|---|---|---|
| `SAM_MODELO` | modelo do Ollama no modo normal | o melhor instalado pra sua placa |
| `SAM_MODELO_LEVE` | modelo do modo leve | `qwen3:4b` (ou `llama3.2:3b`) |
| `SAM_WHISPER` | `cpu` força o ouvido na CPU | `auto` (GPU se tiver) |
| `SAM_WHISPER_MODELO` | tamanho do ouvido na GPU | `medium` com 12 GB+, senão `small` |
| `SAM_CTX` | memória de conversa do modelo (menor = menos VRAM) | `8192` |
| `SAM_VRAM_MB` | finge outra placa (testar a escolha automática) | — |
| `SAM_DADOS` | onde guardar vozes e cache do Whisper | `./dados` |
| `SAM_PORTA` | porta da página | `8770` |
| `OLLAMA_HOST_URL` | Ollama em outro lugar (WSL, outra máquina) | `http://127.0.0.1:11434` |

## Privacidade

- Áudio, transcrições e conversas **não saem do seu computador**.
- Seu nome e o "sobre você" ficam só no navegador (localStorage).
- As sessões salvas ficam na pasta `sessoes/` (fora do git).

## Feito com

[faster-whisper](https://github.com/SYSTRAN/faster-whisper) (MIT) · [Ollama](https://ollama.com) (MIT) · [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (Apache 2.0) via [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx) (MIT) · modelos [Qwen3](https://huggingface.co/Qwen) (Apache 2.0) / [Llama 3.2](https://www.llama.com), cada um com sua licença.

## Licença

[MIT](LICENSE) © 2026 Bruno Castiglioni. Use, modifique e compartilhe.

---

### English

**BCast Language Teacher** is a free, fully local English and Spanish speaking partner built for Portuguese speakers: push to talk, get a spoken reply and instant corrections with explanations in Portuguese. Whisper (speech-to-text) + any Ollama LLM + Kokoro TTS, no cloud, no API keys. Install Python and Ollama, run `ollama pull qwen3:4b` (8 GB GPUs or CPU-only; `qwen3:8b` for 12 GB+), then `instalar` → `iniciar`.
