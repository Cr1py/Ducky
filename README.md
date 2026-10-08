# Ducky

Ducky is a duck "tool" designed to help developers better organize and work through their thoughts. The idea comes from the traditional “rubber duck debugging” technique, where developers explain a problem out loud to clarify their thinking, except Ducky can respond with prompts to lead a developer to a solution. Ducky uses Python, Typer, SQLite, Pydantic, pytest, and Ollama as its local LLM, with a state machine and prompt builder controlling conversation flow and progressively guiding users through questions and hints rather than immediately giving them the answer. It stores conversation history and context in SQLite, allowing developers to start sessions, continue previous discussions, and refine their thinking as they work through a problem.

## What Does it Do??
1. run `ducky thoughts` to let ducky listen to you as you explain your problem.
2. If you realize what's wrong, GREAT ducky did it's job (just without the cool tech part).
3. If not, then Ducky will process your audio into a transcript and feed it into an LLM in order to generate a response.
4. Ducky will then take that response, save it into a session, and then feed it back to the user in the terminal.
5. The user can then decide to end the chat `ducky end`, continue the chat `ducky thoughts`, or move to an older chat `ducky thoughts --session [session name]`. 

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+ (uv can install Python for you).

```bash
uv sync                                      # creates .venv, installs deps from uv.lock
uv run ducky thoughts                        # begin pouring your heart out to ducky
uv run ducky thoughts --text "Example text"  # if you can't speak, then you can type your heart out to ducky
uv run ducky history                         # spits out your current chat history
```

To use `ducky` anywhere without the `uv run` prefix, install it as a tool
(editable, so code changes apply immediately):

```bash
uv tool install --editable .
ducky --help
```

Piping works too: `echo "my loop never ends" | ducky thoughts`.

Managing dependencies:

```bash
uv add rapidfuzz             # add a runtime dependency
uv add --dev ruff            # add a dev dependency
```

## Set up an LLM

Models live in a registry (`src/ducky/models.toml`): local Ollama, Claude, ChatGPT and
Gemini out of the box. Pick one and give Ducky a key:

```bash
ducky config models                        # list models, which is active, which have keys
ducky config set agent claude              # choose a model by name
ducky config set-key claude                # prompts without echo, saves to Ducky's keys file
ducky config keys                          # key status per model (values are never printed)
ducky config show                          # the resolved provider + model ID
```

**Keys** are looked up in this order: real environment variables (ex: `ANTHROPIC_API_KEY`),
then `keys.env` in Ducky's config directory. A `.env` in your current directory is
deliberately **never** read, so another project's keys can't leak in.

**Adding or changing models:** don't edit the packaged file. Create `models.toml` in Ducky's
config directory (`ducky config models` prints the path). Entries are merged field by field:

```toml
[models.claude]                     # override just the model ID
model = "claude-sonnet-5-5"

[models.mine]                       # a new model: a local Ollama model, no key needed
provider = "ollama"
model = "llama3.2"
num_ctx = 8192
```

Fields: `provider` (`openai_compat` | `anthropic` | `gemini` | `ollama`), `model`, `api_key_env`
(omit for keyless local servers), `base_url`, `max_tokens` (default 4096; **raise it for
reasoning models**, which spend tokens thinking and can return an empty reply if it's too low),
`token_param` (`max_tokens` or `max_completion_tokens`), `timeout`. Typos in field names are
rejected with a message naming the entry.

### Ollama (local models)

```bash
ollama pull gemma4:e2b                     # the model the built-in `ollama` entry uses, you can change this to any ollama model, but you will need to install it first
ducky config set agent ollama
ducky thoughts
```

Ducky talks to Ollama's **native** `/api/chat`, not its OpenAI-compatible `/v1` endpoint. The
`/v1` endpoint can't set the context window, and Ollama then silently truncates the prompt to
its default (4k tokens on GPUs under 24 GiB), which could eat Ducky's system prompt. Ollama-only
fields in `models.toml`:

- `num_ctx`: context window in tokens. Ducky also prunes history sooner so prompt + reply
  always fit inside it.
- `keep_alive`: how long the model stays loaded after a request (ex: `"30m"`).
- `think`: `false` (or a model-specific string such as `"low"`) for thinking models. Leave it
  unset for models without a thinking mode, which reject the field.
- `timeout`: the built-in entry uses 120s because the first request after a pause loads the model.

Ducky also asks Ollama for schema-constrained JSON, which should make small models more
reliable about the reply format than prompting alone. For **Ollama Cloud**, add an entry with
`base_url = "https://ollama.com"` and `api_key_env = "OLLAMA_API_KEY"` (example in `models.toml`). If you paste an OpenAI-style URL ending in `/v1`, Ducky trims it.

The default model IDs in the packaged file are placeholders that go stale. Check each
provider's docs and override as above.

Gemini uses the stateless `generateContent` endpoint rather than the Interactions API, which
stores interactions server-side by default and would conflict with Ducky's local-first history.

For offline demos and tests: `DUCKY_LLM=stub` uses a fake LLM with no key needed.

## How memory works

- The system description and problem statement are pinned and never pruned.
- The last 6 turns always stay verbatim.
- When the live context passes `max_context_tokens` (estimated ~4 chars/token), older
  turns are folded into a rolling summary. Rows are kept in SQLite; they just stop
  being sent to the LLM. This happens *after* the reply is shown.
- `ducky end` and switching sessions also refresh the summary (skipped for tiny sessions).
- If the summarizer call fails, nothing is lost and it retries next time.

## Commands

| Command | What it does |
|---|---|
| `ducky thoughts [--new] [--name N] [--session N] [--text T] [--skip-intro]` | One turn of talking to Ducky (continues the active session by default). |
| `ducky end` | Save summary, clear the active session. |
| `ducky history [--session N]` | Pinned context, summary, remaining turns. |
| `ducky session list \| rename OLD NEW \| delete NAME [-y]` | Manage sessions. |
| `ducky config set KEY VALUE` / `show` / `models` / `keys` / `set-key MODEL` | Settings: `answers`, `agent`, `silence`, `tone`, `stt`, `max_context_tokens`. |
| `ducky phrases list \| add \| remove \| test` | End phrases (min 2 words). |
| `ducky quack` | Quack. |

## Project Structure

```
src/ducky/
  cli.py          # Typer app (subcommands)
  config.py       # TOML config, validated on write (DUCKY_HOME overrides paths)
  db.py           # SQLite + migrations (PRAGMA user_version)
  sessions.py     # sessions, turns, active-session pointer
  turn_runner.py  # one turn: capture -> save -> LLM -> apply -> save -> prune
  input/          # InputSource: text (done), voice (M4)
  stt/            # STTEngine interface (Vosk in M4)
  registry.py     # models.toml loader/validator (built-in merged with the user's file)
  models.toml     # built-in model registry
  keys.py         # API key lookup (env first, then Ducky's keys file)
  llm/            # schema, prompts, client, errors, adapters (OpenAI-compat / Anthropic / Gemini / Ollama)
  endphrase.py    # tail matching + phrase storage (fuzzy in M5)
  memory.py       # pruning + rolling summary
  render.py       # all Rich output
```

## Considerations
- Ducky needs context in order to maximize how well it can help. The user needs to be descriptive of their problem and current solution (currently, Ducky will try their hardest to prompt the user to be very descriptive if it is unsure on how to approach the problem).
- Expect local models of modest size to be weaker than hosted ones for more context or reasoning heavy jobs.
- Ducky is pretty good at helping with algorithmic problems (think LeetCode), but still needs to be tuned for System Design or logical reasoning in larger code bases. 

## Privacy

Speech-to-text runs locally (Vosk). Only text transcripts are stored, and no audio
is saved. Transcripts are still sent to whichever LLM provider you configure.
