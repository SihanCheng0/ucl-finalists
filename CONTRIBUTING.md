# Working on UCL Finalists

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and Node.js 22 or newer, then:

```bash
make setup   # uv sync, npm ci in web/, and the dashboard build
make test    # everything should pass before you change anything
```

LM Studio is optional. You need it only to write new AI write-ups (see [The local LLM](#the-local-llm)).

## Workflow

1. Branch from `main`: `git switch -c your-name/short-topic`.
2. Make the change with its tests, then run `make test`.
3. Open a pull request. CI runs the Python tests, the frontend tests and the frontend build, and a teammate
   reviews before merging.

Keep pull requests small and about one thing. Commit messages follow the existing history: a type, then what
changed in plain words, such as `fix: squads that page past their end` or `docs: how the cache works`. The
types used are `feat`, `fix`, `docs`, `test` and `refactor`.

## Tests

| Command | Runs |
|-|-|
| `make test` | Both suites below |
| `uv run pytest -q` | The Python suite in `tests/` (about 15 seconds) |
| `uv run pytest tests/test_analyst.py -q` | One file |
| `npm --prefix web test` | The frontend tests (Vitest), in `web/src/lib/*.test.ts` |

Tests never touch the network or LM Studio: UEFA and the LLM are replaced with fakes (`tests/fakes.py`,
`tests/synthetic.py`). Keep it that way, so the suite stays fast and runs in CI.

**Golden files.** `tests/test_cli_characterization.py` compares the CLI's output with `tests/golden/cli/*.txt`. When
you change that output on purpose, record it again and review the diff before committing:

```bash
UPDATE_GOLDEN=1 uv run pytest tests/test_cli_characterization.py -q
git diff tests/golden
```

## Code style

- **Python 3.13:** type hints, small functions, and docstrings that say why rather than what. Lines stay within
  about 120 characters. Match the code around your change.
- **TypeScript and React:** strict TypeScript. `npm run build` type-checks before it bundles. Logic that can be
  tested without a browser goes in `web/src/lib/`, with a test beside it.
- **Words on screen** are plain and specific, in sentence case. An error says what went wrong and how to fix it.

## The data

Everything the pipeline reads and writes is committed, so everyone sees the same numbers.

| Path | Rule |
|-|-|
| `data/raw/` | UEFA's responses, exactly as sent. Never edit by hand. `uv run ucl fetch` fills in what's missing |
| `data/processed/`, `out/` | Pipeline outputs. Regenerate with `uv run ucl build`, `model`, `analyze` or `report`, never by hand |
| `data/llm_cache/` | Saved LLM answers. Commit new ones together with the `out/analysis.json` they produced |

When a code change alters the outputs, rerun the stages it affects and commit the new outputs in the same pull
request, so the numbers in `out/` always match the code. If you ran the pipeline only to try something,
`git restore data out` puts the committed files back. Large data files are marked in `.gitattributes`, so GitHub
collapses them in diffs.

## The local LLM

The analyze stage uses `qwen/qwen3.8-27b` in [LM Studio](https://lmstudio.ai/), with a 16,384-token context.
Every answer is saved in `data/llm_cache/` under its exact request, and analyze tries the saved answers first. When
every question has been asked before, it rebuilds the write-ups without LM Studio.

A new question, from changed facts or a different model, needs LM Studio:

1. Install LM Studio and open it once, then run `~/.lmstudio/bin/lms bootstrap` so the `lms` command works.
2. Download the model: `lms get qwen/qwen3.8-27b` (about 16 GB). It needs LM Studio's MLX engine 1.11 or newer:
   `lms runtime update mlx` updates it. Older engines fail to load it with "Unrecognized image processor".
3. Run `uv run ucl analyze`. It starts LM Studio's server and loads the model with the right context if needed.

In UCL Lab, **Before you run** on the Pipeline screen checks each of these, and **Test the local model** asks the
model one question to prove it answers. `uv run ucl all --no-ai` runs everything except the write-ups.
