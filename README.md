# PoliticalGPT

What does CNN's writing voice sound like versus Fox's, when they're describing the *exact same made-up event*?

PoliticalGPT fine-tunes two [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) models via QLoRA — one on CNN political coverage, one on Fox News political coverage — then generates paired articles about identical, entirely fabricated events and slides continuously between the two outlets' framing of them. It's part demo, part research artifact: the same invented headline, run through two different newsroom "personalities," with the framing divergence measured rather than just eyeballed.

## The idea, in one sentence

Real CNN and Fox coverage differs in topic, timing, sourcing, and a dozen other things that have nothing to do with framing — so instead of comparing real articles, this project invents a neutral event, feeds the identical prompt to both fine-tuned models, and only the outlet's absorbed writing style is left to vary.

That's the whole trick. Everything else in this repo exists to make that trick rigorous.

## What it actually does

1. **Two fine-tunes, one base model.** A shared, pinned Llama-3.1-8B-Instruct is fine-tuned twice via QLoRA — once per outlet — on byte-identical training configs, so any difference in output is attributable to the training data, not to some stray hyperparameter.
2. **A continuous framing dial.** Rather than picking one adapter or the other, both LoRA adapters stay loaded simultaneously and get blended at inference time (λ = 0 is pure Fox, λ = 1 is pure CNN, and everything in between is a genuine interpolation — not a coin flip between two frozen outputs).
3. **A mad-libs event generator.** Pick a politician, an action, a setting, a policy area, and an outcome from dropdowns — no free text — and the dial generates a fabricated but structurally plausible headline + article in whatever blend of framing you've dialed in.
4. **An offline measurement layer.** Framing divergence across many generated pairs gets scored along a few literature-grounded axes (arousal, lexical/informational bias, power-and-agency framing), so the "CNN sounds more X, Fox sounds more Y" claim isn't just vibes.

Every generated article — in the demo and everywhere else — carries a persistent **"AI-generated / this event did not happen"** label. Nothing here is a real news story about a real event, and it's never meant to be mistaken for one.

## Why this exists

This project is a direct sequel to a university Text Information Systems project that quantified 20 years of headline sensationalism through observational analysis of real articles. This version asks the same underlying question — *how does outlet framing actually manifest?* — but answers it through controlled counterfactual generation instead: hold the event fixed, vary only the outlet, and measure what changes. It's built to hold up as a legitimate research artifact, not just a fun side project (though it is also, unapologetically, a fun side project).

## Status

This is being built in phases, roughly in dependency order:

| Phase | What it delivers |
|---|---|
| Preregistration & contracts | Analysis plan, prior-art positioning, and shared config/prompt contracts — locked before any result exists that could bias them |
| Dial correctness | Proving the λ-interpolation math is exact, on CPU, before any GPU money is spent |
| Corpus construction | Two outlet corpora whose only systematic difference is framing |
| Fine-tuning + safety gates | Training both adapters and clearing an off-domain alignment check before either is used downstream |
| Generation harness | Producing every experimental generation once, seeded and logged |
| Measurement layer | Scoring the framing divergence and computing the headline result |
| Demo | The actual interactive dial you can play with |

Check back — or check the commit history — for where things currently stand.

## A few things worth knowing

- **Real politicians, fake events.** The people are real; the events are not. This is a deliberate choice, made with eyes open to the tradeoffs, and mitigated by mandatory labeling on every output.
- **No in-app bias scores.** The measurement layer is a research artifact reported in aggregate, not a live number stamped on each generated article — a per-article score would be read as more authoritative than it deserves to be.
- **No free-text prompts.** The mad-libs interface is a deliberate constraint, not a missing feature — it's what keeps the comparison controlled.

## Tech

Python, Hugging Face `transformers` + `peft` + `trl` for QLoRA fine-tuning and dial mechanics, with a FastAPI backend and a React frontend for the interactive demo.

---

*Built by [Vinay Panayanchery](https://github.com/vinay-pan). Not affiliated with, endorsed by, or representative of CNN, Fox News, or any political figure depicted.*
