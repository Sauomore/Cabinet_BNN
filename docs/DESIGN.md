# Design & Research Direction

[← Back to README](../README.md) ｜ [中文](DESIGN.zh.md)

The idea underneath the implementation, the measured facts that constrain it,
and the open problem being worked on.

## Research direction

Measured fact that motivates the current work:

> An **independently randomised** `param` segment **destroys** the weight geometry:
> `corr(hash Hamming distance, weight cosine similarity) = +0.0019` (i.e. none).
> Using only the semantic `mask` gives `−0.9885` (near-perfect monotonicity).

Interpretation: a single `k`-dimensional sign vector **cannot** simultaneously carry the
*geometry* and provide *independent degrees of freedom*.

The direction being explored instead is **code-as-index**:

```
code (hash)     →  index        →  geometry is preserved by construction (locality of addressing)
information code →  content      →  free to hold anything; needs no geometric constraint
```

together with **MoE-style sparse activation**, so that editing one token's information
code touches only the `k'` basis matrices it actually activates — i.e. sparse writes, not
just sparse reads.

**Open problem** (this is where the author believes the elegant answer lies):

> Can a set of shared basis matrices be learned such that editing any single code has a
> **bounded** effect on all other tokens' outputs?
>
> If yes, this would be a parameterisation-level formulation of the
> stability–plasticity trade-off — and locality is exactly the mechanism by which
> biological synapses (STDP) achieve continual learning without global gradients.

## What counts as done — acceptance criteria

The project's north-star metric is **cost-adjusted parity**, not superiority:

> The goal is not to beat existing methods. It is to show that per-token weight
> codes are **not worse** — and cheaper, exactly reversible, and usable as a
> retrieval key at the same time.

This mirrors how [BitNet b1.58](https://arxiv.org/abs/2402.17764) positions
itself: matching quality, then reducing cost. Parity is a publishable result when
it comes with a cost reduction; parity at equal cost is only "another
implementation" and is not a contribution.

#### Self-learning feasibility, stated so it can be measured

"Self-learning works" has to be a checklist, not a feeling. The claim is
**feasibility**, and it is decomposed into four verifiable conditions:

| # | condition | status | evidence / how to measure |
|---|---|---|---|
| ① | After adding a domain, old-domain performance does not drop | ✅ **measured** | per-domain code tables: forgetting **0.0000**, bit-exact rollback |
| ② | Adaptation cost is lower than LoRA at matched quality | ⬜ pending | 250 KB/domain vs LoRA r=16 500 KB — cost is known, **quality parity is not** |
| ③ | Tokens can be added or edited individually, without retraining the backbone | ⬜ pending | sparse write is implemented (blast radius `k'/k`, 0 other tokens affected); no end-to-end test yet |
| ④ | It holds on a real task, not only on 5 synthetic domains | ⬜ pending | the ① result is from 3,109 words / 5 synthetic domains |

**At the end of the project every row is either ticked or crossed, with the
measurement attached.** No row is settled by impression.

#### The comparison that decides ②

Quality parity is the precondition, so it must be measured first. Then, **given
matched quality**, compare:

| axis | this project | LoRA r=16 |
|---|---|---|
| storage per domain | 250 KB | 500 KB |
| rollback granularity | **bit-exact** | file-level replacement |
| code doubles as retrieval key | **yes** (HSH-64, one `popcnt`) | no |
| quality | **to be measured** | baseline |

If quality matches and the other three columns favour this project, the claim is:

> **a per-domain adaptation scheme that is cheaper, exactly reversible, and
> simultaneously serves as a retrieval index.**

That is defensible and survives review, because it is quantitative. A claim of
"we solved catastrophic forgetting" would not — zero forgetting here comes from
**physical isolation** (the same mechanism as PackNet / HAT / PNN), and is paid
for with capacity.


---

[← Back to README](../README.md) ｜ [中文](../README.zh.md)
