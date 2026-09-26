# Project description (GitHub "About" field)

GitHub's About field allows **350 characters** and shows roughly the first 100 in
search results, so the opening clause has to stand on its own. Everything below
is written to be pasted directly; character counts include spaces.

## Recommended

    A language model whose per-token weights are generated from a 64-bit semantic
    code instead of a global weight tensor. Binary activations, bit-level
    reversible adaptation, and negative results reported honestly.

`241 characters` — leads with what it *is*, names the three searchable ideas
(per-token weights, semantic code, binary), and ends on the differentiator.

## Alternatives

**Shortest** — if you prefer a terse line:

    Per-token weights generated from a 64-bit semantic code. Binary activations,
    HSH-64 retrieval, honest ablations.

`115 characters`

**Research-framing** — foregrounds the question rather than the artifact:

    Can a token's weight matrix be *generated from* a compact discrete code
    rather than looked up? Binary activations + HSH-64 semantic codes, with
    controlled ablations including negative results.

`191 characters`

**Emphasis on editability** — if the self-learning angle matters more:

    Binary-activation LM whose per-token weights come from writable 64-bit
    semantic codes — enabling exact-reversible domain adaptation and sparse,
    local parameter edits.

`172 characters`

## Topics (the tag row under About)

These drive GitHub search far more than the description text does. Recommended,
in priority order:

    language-model  binary-neural-network  quantization  semantic-hashing
    continual-learning  mixture-of-experts  pytorch  hashing  lsh
    parameter-efficient  hsh-64  weight-generation

Note: `pytorch` and `language-model` are high-traffic tags; `hsh-64` and
`weight-generation` are near-empty, so the repository can rank on them
immediately.

## What **not** to put in the description

- **Do not claim superiority.** "Outperforms LoRA" and "solves catastrophic
  forgetting" are both contradicted by measurements in this repository. Anyone
  who reads `docs/RESULTS.md` will find the controlled comparisons, and an
  overstated About field costs more credibility than it buys attention.
- **Do not say "AGI" or "brain-like".** Binarisation buys cost, not intelligence,
  and the relation gate — the most "brain-like" part — was measured to hurt.
- **Keep the model size out of it unless the number helps.** "41.6 M parameters"
  is honest but invites a size comparison the project does not win.

## Website field

If a link is wanted, point at the results document rather than the repository
root, so a visitor lands on the measured numbers:

    https://github.com/Sauomore/Cabinet_BNN/blob/main/docs/RESULTS.md
