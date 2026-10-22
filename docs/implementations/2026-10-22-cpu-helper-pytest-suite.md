# CPU helper pytest suite

## Problem / Motivation

[Issue 12](https://github.com/Cassandra-Collective/CASSANDRA/issues/12) asks
for fast CPU coverage of the helpers that carry the lab's methodology. Syntax
checks and training smoke runs cannot independently verify count tables,
answer-position selection, codec mappings, or seeded corpus contracts.

## What Changed

| File | Change |
| --- | --- |
| `tests/conftest.py` | Adds the lab directory to the import path for sibling imports. |
| `tests/test_transformer_helpers.py` | Tests smoothed count logits, codecs, split margins, verifier modes and offsets, and the zero-residual frozen-prior behavior. |
| `tests/test_flagship_eval_helpers.py` | Tests ASCII encoding and alphabet normalization without loading checkpoints. |
| `tests/test_corpus_generators.py` | Tests seeded corpus output, copy answers, held-out key separation, transition metadata, and invalid parameters. |
| `experiments/tiny_language_lab/README.md` | Documents the test location, CPU dependencies, command, and scope. |
| `CONTRIBUTING.md` | Replaces the outdated no-test-suite statement with a link to the test instructions. |
| `docs/implementations/2026-10-22-cpu-helper-pytest-suite.md` | Records the approach, design decisions, and verification. |

## Implementation Approach

The tests import the existing scripts directly. No research implementation,
CLI defaults, role files, or recorded results change. Corpus builders run only
in memory with small inputs. PyTorch tensors and the tiny model use the CPU;
the model has no dropout, and its seeded construction restores the caller's
torch RNG state. Corpus checks also verify that the global Python RNG state
is unchanged.

Expected count entries come from an explicitly enumerated sequence of
transitions. Verifier cases use independently specified offsets, cover LF
and CRLF, and distinguish identity mode from key-answer mode. The latter
intentionally accepts nonidentity mappings; equality is checked only in
identity mode. Split examples use lengths for which both sides can hold the
required window. Encoding and normalization examples include explicit codec
order, Unicode decomposition, unsupported input, empty input, and space runs.

The stretch check builds a tiny transformer, applies the same
`zero_output_head` helper used by the count-bigram training path, and checks
exact equality between its initial logits and the frozen table indexed by
the input tokens. It performs no optimization and creates no checkpoint.

## Mathematical / Statistical Details

For the count-table example, the sequence is `0, 1, 0, 1, 2, 2`. Transition
`0 -> 1` occurs twice; `1 -> 0`, `1 -> 2`, and `2 -> 2` occur once. Each
cell receives smoothing mass `0.5` before taking its natural logarithm.
These are unnormalized logits, not row-normalized probabilities.

Markov transition rows must sum to one within the rounding tolerance of the
metadata serializer. The bits-per-character metadata must equal the value
in natural-log units divided by the natural logarithm of two. No statistical
hypothesis test or research performance claim is introduced.

## Design Decisions

- Use a root `tests/` directory and a small `conftest.py`, as the issue asks,
  rather than moving scripts or introducing a package.
- Use pytest parametrization for boundaries, verifier modes, invalid inputs,
  and fixed seeds. Each test docstring identifies its behavioral contract.
- Exercise actual helpers without mocking torch, corpus generation, or model
  behavior. No downloads or external services are involved.
- Keep pytest as the only new testing dependency. Torch and NumPy are
  existing lab dependencies; packaging and CI remain separate issues.
- Document the command in the lab README, leaving the owner's role files
  unchanged.

## Verification

From the repository root, in a Python environment with the CPU dependencies
shown in the lab README:

```powershell
python -m pytest
python -m pytest -q
python -c "import ast, pathlib, subprocess; files = [pathlib.Path(p) for p in subprocess.check_output(['git', 'ls-files', '*.py'], text=True).splitlines()]; [ast.parse(p.read_text(encoding='utf-8'), str(p)) for p in files]; print(len(files), 'files parse')"
git diff --check
```

Validation on `2026-10-22` used the Desktop checkout through local WSL
Linux, Python `3.14.7`, pytest `9.1.1`, NumPy `2.5.4`, and torch
`2.14.1+cpu`. Torch reported no CUDA build and no available CUDA device.

- `python -m pytest -q --durations=5`: `76 passed in 4.80s`.
- A local validation harness repeated the full suite with Python audit
  hooks rejecting socket connection and DNS requests. With hash seeds `0`
  and `123`, it reported `76 passed in 1.26s` and `76 passed in 0.70s`,
  respectively, with zero network attempts.
- Nine deliberate in-memory mutations were detected: ignoring either
  corpus seed, leaking held-out keys, reversing count transitions,
  skipping identity equality, accepting the excluded limit, returning
  incorrect encoding IDs, omitting lowercase normalization, and leaving
  a nonzero residual bias. Each selected test group failed as intended.
  No research source was modified for these checks.
- All `48` tracked Python files, including the tests, passed AST parsing.
- The diff whitespace check and changed-Markdown dash check passed. No
  generated corpus, checkpoint, dataset, or run output was staged.

The existing quick-start commands also completed on CPU in a visible
PowerShell window, invoking the same local WSL environment:

```powershell
python experiments/tiny_language_lab/make_synthetic_corpus.py --lines 320 --seed 20260616
python experiments/tiny_language_lab/cassandra_tiny_lm.py --method count --device cpu --corpus experiments/tiny_language_lab/corpus/structured_seed.txt
python experiments/tiny_language_lab/cassandra_tiny_transformer.py --device cpu --steps 20 --eval-interval 10 --max-new-tokens 80 --corpus experiments/tiny_language_lab/corpus/structured_seed.txt
```

The corpus generator reported `25,586` characters. The bigram command
completed with exit code `0`; the transformer completed all `20` steps
with exit code `0`. Both ran with `OMP_NUM_THREADS=1` and
`MKL_NUM_THREADS=1`. These are compatibility checks, not new research
comparisons, and existing results were left untouched.

Native Windows Python `3.11.9` passed the corpus module (`26 passed in
0.03s`), but Smart App Control blocked the installed `torch_python.dll`
(`WinError 4551`) during full-suite collection. Full-suite evidence above
is from Linux on this same machine, not native Windows. No Windows
security setting was changed.

## Related Docs

- [Issue 12](https://github.com/Cassandra-Collective/CASSANDRA/issues/12)
- [Contributing guide](../../CONTRIBUTING.md)
- [Lab README](../../experiments/tiny_language_lab/README.md)
- [CLAUDE.md: count prior and verifier behavior](../../CLAUDE.md)
- [CODEX.md: evidence standard](../../CODEX.md)
