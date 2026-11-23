# Browser World Model v0 (BW0): locked specification

**Status:** locked for the one-day build. Change only with a measured reason.
**Use case:** appointment booking. **The capability being tested:** predicting what an action will do, several steps ahead, before acting.

---

## 1. The goal, reduced to what can be checked

> Given what the browser shows now and a proposed sequence of actions, predict what the page will
> look like after each action, without touching the browser. Do it faster than an LLM, on booking
> forms the model has never seen, and use those predictions to finish a booking with fewer real
> observations than an agent that looks at the page after every step.

Each step in this document must move one of these four numbers:

| # | Metric | Measured against |
|---|---|---|
| 1 | Accuracy on changed elements (F1), 1/3/5 steps ahead, on unseen layouts and wording | no-change baseline, prompted Qwen |
| 2 | Time per imagined step | prompted Qwen on the same input |
| 3 | Real observations per completed booking | agent that observes after every step |
| 4 | Booking success rate and failed submits | same agent |

---

## 2. Where we reuse proven tools vs. design from first principles

**Reuse (about 80% of the work, nothing new invented):**

| Part | Proven tool or pattern |
|---|---|
| Environment server | FastAPI or Flask + plain HTML/JS |
| Browser | Playwright + Chromium (already in the container) |
| Observation | Our existing four-plane observer (semantic, geometry, motion, temporal), ported to Python Playwright's CDP session |
| Label embeddings | Frozen sentence encoder (MiniLM-class), cached once per unique string |
| Model structure | Latent dynamics with decoders, as in Dreamer / TD-MPC2: encoder → latent → dynamics → heads |
| Training | PyTorch, cross-entropy / binary cross-entropy, AdamW |
| Planning | Exhaustive or beam search over a small action space (as in chess) |

**First principles (only the bottlenecks):**

1. **What exactly is predicted.** Most elements don't change after an action (estimated over 90% in
   form steps). A model scored on the whole page learns to copy. So the targets are **per-element
   changes**, and the loss is **weighted toward elements that changed**. Copy-everything must score badly.
2. **Where typed values come from.** A field's value has only four possible sources: it stays the
   same, it takes what was typed, it takes the selected option, or it's cleared. So value prediction
   is a **4-way choice of source**, not text generation. This is exact, fast, and can't produce
   made-up values.
3. **Validation without writing the rules into the model.** The model must learn that `john@x` is
   invalid. If we add a hand-written `is_valid_email` feature, we've coded the rule ourselves and the
   claim that it learned anything is gone. So typed text goes in through a small
   **character-level encoder**, and validity is learned only from observed outcomes.
4. **Element identity across steps.** The observer uses CDP `backendNodeId`s, which can change when
   the page re-renders or navigates. Labels need correct before/after matching. Use the environment's
   `data-wam-key` attributes **for label matching only, never as model input** (real sites won't have
   them). If those are missing, fall back to matching on (role, name, DOM path).
5. **When to look at the real page.** Every action in a booking form is either **reversible**
   (typing, selecting, checking a box) or **a point of no return / unknown** (submit, navigation to a
   page not seen yet). Imagine and execute reversible stretches without looking. Observe **only**:
   after navigation (the new page can't be predicted), right before an irreversible action, and
   after it. A rule-based policy, so it needs no calibration on day one.
6. **Compute.** A dynamics model of 1–3M parameters on about 20–50k transitions trains in minutes
   on a CPU. **No GPU is on the critical path.** Kaggle is used only for the optional prompted-Qwen
   baseline.

---

## 3. What is in and out of scope

| In v0 | Out of v0 (with the reason) |
|---|---|
| Accessibility semantics, geometry, motion/layers, temporal signals | Screenshots and a vision encoder: we own the environment, so it can be fully accessible. A vision ablation is planned for v1. |
| Learned latent state + recurrent memory | EMA / JEPA latent targets: the decoders already supervise the latent, so they aren't needed |
| Multi-step latent rollout training (K=1 vs K=3) | Learned policy head: the action space is small, so search is better |
| Search planner + rule-based observation checkpoints | Calibrated confidence controlling when to observe: v1 |
| Predicting new elements as typed events (alert, option list, panel) | Predicting a whole page it hasn't seen: re-observe after navigation instead |
| Prompted Qwen baseline | Fine-tuning Qwen (QLoRA): only if BW0 fails on unseen wording |
| | Weight-space programming: a separate research track |

**Known limit, stated honestly:** information that exists only visually is invisible to BW0. That's
acceptable here only because the environment is ours.

---

## 4. Environment (E0.1)

A Python backend holds the booking rules and state; a minimal HTML/JS frontend handles native
browser interaction. **State is kept per session** (one session id per browser context), so parallel
workers can't interfere with each other. **The clock is fixed** per scenario. `POST /reset` restores the
scenario exactly.

Behavior that must pass tests before anything else is built:

| Action | Required real result |
|---|---|
| Select service | Available doctors/dates update |
| Select date | Time slots update (unavailable slots shown as `aria-disabled`) |
| Invalid email + submit | Validation error, no booking created |
| Valid details + confirm | Booking created on the server |
| Book the same slot again | Conflict error |
| Reset | Original availability restored |
| Next → Back | Entered values persist |

Build it deterministic first. **Then** add variation from a scenario config: layout (single page /
2–4 step wizard), field order, label wording, which fields exist, validation rules, availability.

---

## 5. Observation and data (E0.2–E0.4)

**Observer:** port the existing four-plane observer to Python (`context.new_cdp_session(page)`, using
the same CDP calls). Viewport culling stays on. The text plane stays off.

**Leakage rule:** anything measured *after* the action (paint/commit counts, timing, new element
IDs) is a **target only**, never an input.

**Raw record (immutable, one per transition):** schema version, episode id, step, parent branch id,
scenario config (never used as input), full `before` observation, `action`, full `after`
observation (settled, with a settle flag and timeout flag), backend verification (booking created?
outcome type), timing.

**Collectors:** correct solver, noisy solver (deliberate mistakes), random valid actions. The mix starts
at 40/30/30 and is adjusted using the coverage report (persistence, locality, dependency, validation,
affordance, navigation, conflict).

**Branching:** reset backend and browser → replay the action prefix → **check that the
observation matches the branch point** → take an alternative action. Branches share a parent id.

**Quality gates (automated):** action hit the intended element; page settled or flagged; IDs and
boxes valid; after-state belongs to this episode; backend agrees with the recorded outcome; replay
reproduces; synthetic data only.

**Splits, fixed before training:**
- **A:** unseen values
- **B:** unseen layouts and wording
- **C:** unseen combinations of behaviors

**Volume:** start training once there are ≥300 clean episodes; target 2–5k episodes if measured
throughput allows.

**Storage:** `data/raw`, `data/curated`, `data/manifest.json` (commit hash, seeds, counts, hashes),
`artifacts/`. All git-ignored.

---

## 6. Model (M0.1–M0.2)

```
element token  = role_emb + proj(label_emb[frozen, cached]) + charCNN(value) + flags + box + visible/layer
page token     = proj(text_emb(url path / title)) + step index
action token   = type_emb + charCNN(typed text / option); added to its target element's token

encoder   : 2-layer set transformer over element tokens       → element latents z_t
memory    : GRU over pooled latents                            → h_t (hidden wizard state, history)
dynamics  : 2–3 transformer layers over [z_t, action, h_t]     → z_{t+1}, h_{t+1}
heads     : per element  changed? · next flags · value source {keep, typed, option, clear} · removed?
            events       new alert / options / panel (+ which element it's attached to, message class)
            page         navigates? · outcome {none, validation_error, conflict, success}
```

Size: about 1–3M trainable parameters. Loss: a sum of cross-entropy / binary cross-entropy terms, with
elements that changed weighted up. Multi-step: unroll K steps in latent space along real action
sequences, with heads supervised at each step (same as Dreamer / TD-MPC2). This is cheap for a tensor
model, so we train **K=1 and K=3 and compare them** (an ablation).

Message text for new alerts is a class learned from training messages. Unseen message wording counts
as a known failure and is reported, not hidden.

---

## 7. Planner and demo (D0.1)

- **Action space:** for each field, type the matching profile value; select each visible option;
  click each button. Small enough for beam search over imagined rollouts.
- **Score:** P(success) − penalty × P(validation error or conflict) − 0.01 × steps.
- **Execute** the best reversible stretch as a batch; observe at checkpoints (§2.5); never submit
  without a real observation first.
- **Demo:** the prediction is shown on screen *before* the action, then the real result appears next to it.
  Report metrics 1–4 against the baselines.

---

## 8. Day plan and kill criteria

| Gate | Work | Budget |
|---|---|---|
| E0.1 | Environment + tests | 2 h |
| E0.2 | Port observer, identity matching, leakage rule | 1.5 h |
| E0.3–4 | Recorder, collectors, branching, quality gates, splits | 2 h (+1 h collection in background) |
| M0.1 | Model + training (CPU) | 2 h build, under 30 min run |
| M0.2 | Multi-step evaluation, K=1 vs K=3, no-change baseline | 1 h |
| (optional) | Prompted Qwen baseline on Kaggle, run in parallel | 1 h |
| D0.1 | Planner + live demo vs observe-every-step agent | 2 h |
| — | Buffer | 1.5 h |

**Kill criteria (stop rather than fake a result):**
- Fewer than 300 clean episodes by hour 7: skip the planner and ship the dataset + evaluation.
- Doesn't beat the no-change baseline on changed-element F1 for split B: stop and diagnose; no demo.
- 3-step accuracy falls below a useful level (to be set after M0.1): the demo shows 1-step only.
