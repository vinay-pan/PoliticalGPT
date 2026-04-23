# PoliticalGPT — Implementation Plan

## Context

The goal is to build a web app that fine-tunes Llama 3 8B twice — once on CNN articles, once on Fox News articles — both covering Trump/Biden/Obama/Hillary/Kamala from 2015–2024. The app exposes two modes:

1. **Mad Libs generator**: User picks [politician] [action verb] [object] from dropdowns → both models generate a fake news article in parallel, shown side by side.
2. **Q&A chat**: User asks a political question → both models answer, shown side by side.

The bias contrast between the two outputs is the core of the experience.

---

## Architecture Overview

```
PoliticalGPT/
├── data/
│   ├── scraper/
│   │   ├── cnn_scraper.py       # Scrapes CNN politics articles
│   │   ├── fox_scraper.py       # Scrapes Fox News politics articles
│   │   └── utils.py             # Shared: rate limiting, dedup, storage
│   └── raw/                     # JSONL files: {source, date, url, text}
├── training/
│   ├── prepare_dataset.py       # Cleans, filters, formats to instruction JSONL
│   ├── train.py                 # Shared LoRA trainer (takes --source cnn|fox)
│   ├── requirements.txt
│   └── notebooks/               # Colab/RunPod notebooks for training runs
├── api/
│   ├── main.py                  # FastAPI app
│   ├── inference.py             # Loads base model + swaps LoRA adapter
│   ├── prompts.py               # Prompt templates for madlibs vs Q&A
│   └── requirements.txt
└── frontend/
    ├── src/
    │   ├── components/
    │   │   ├── MadLibs/
    │   │   │   ├── MadLibs.tsx       # 3 dropdowns + generate button
    │   │   │   └── ArticlePane.tsx   # Side-by-side article display
    │   │   ├── QAChat/
    │   │   │   ├── QAChat.tsx        # Text input + submit
    │   │   │   └── AnswerPane.tsx    # Side-by-side answer display
    │   │   └── ui/
    │   │       ├── SourceBadge.tsx   # CNN vs Fox badge styling
    │   │       └── LoadingSpinner.tsx
    │   ├── App.tsx                   # Tab switching between modes
    │   └── styles/
    │       └── tokens.css
    └── package.json
```

---

## Phase 1 — Data Collection

### Scrapers

**CNN** (`data/scraper/cnn_scraper.py`):
- Target: `cnn.com/politics` and search pages for each politician name
- Use `requests` + `BeautifulSoup` for static pages; `playwright` for JS-rendered pages
- Filter: article must mention at least one target politician by name
- Target: ~3,000–5,000 articles per politician (15k–25k CNN total)

**Fox News** (`data/scraper/fox_scraper.py`):
- Target: `foxnews.com/politics` and `/category/person/*`
- Same tech stack and filtering criteria
- Target: matching volume to CNN (~15k–25k articles)

**Shared utils** (`data/scraper/utils.py`):
- 1–2s random delay between requests (rate limiting)
- URL deduplication via SQLite or a seen-set
- Output: JSONL per source, one record per article: `{source, url, date, title, text, politicians: []}`

---

## Phase 2 — Dataset Preparation

**`training/prepare_dataset.py`**:
- Input: raw JSONL files
- Filter: articles 200–2000 words, date 2015–2024, at least one politician mention
- Format each article into instruction-tuning format:

```
### Instruction
Write a news article about {politician} in the style of {source}.

### Response
{article_text}
```

- Output: `data/cnn_train.jsonl` and `data/fox_train.jsonl`
- Split: 90% train / 10% eval

---

## Phase 3 — Fine-Tuning

**`training/train.py`** (run twice: `--source cnn`, `--source fox`):
- Base model: `meta-llama/Meta-Llama-3-8B-Instruct`
- Method: QLoRA (4-bit NF4 quantization via `bitsandbytes`)
- Library: Hugging Face `transformers` + `peft` + `trl` (SFTTrainer)
- LoRA config: `r=16`, `alpha=32`, target modules: `q_proj`, `v_proj`
- Training: ~3 epochs, batch size 4, gradient accumulation 4
- Hardware: RunPod A40 (48GB VRAM) or 2x A6000 — estimated 4–6hrs per model
- Output: two adapter directories: `models/cnn-adapter/` and `models/fox-adapter/`

---

## Phase 4 — Inference API

**`api/inference.py`**:
- Load Llama 3 8B base model once at startup (quantized)
- Keep both LoRA adapters loaded; swap via `model.set_adapter()`
- Both CNN and Fox responses generated in parallel (two async tasks)

**`api/prompts.py`** — two prompt builders:
- **Mad Libs**: `"Write a {source}-style news article about {politician} who {verb} {object}."`
- **Q&A**: `"Answer this political question in the style of {source}: {question}"`

**`api/main.py`** — FastAPI endpoints:
```
POST /generate/madlibs
  body: { politician, verb, object }
  response: { cnn: string, fox: string }

POST /generate/qa
  body: { question: string }
  response: { cnn: string, fox: string }
```

---

## Phase 5 — Frontend

**Stack**: React + TypeScript + Vite, plain CSS with custom properties (no component library — avoid template look)

**Visual direction**: Satirical / tabloid aesthetic. Split screen with hard left/right branding (CNN blue vs Fox red). Typography with personality. Not a generic card grid.

**`App.tsx`**: Tab bar switching between "Fake News Generator" and "Ask Both Sides"

**Mad Libs mode** (`MadLibs.tsx`):
- Dropdown 1: Obama / Hillary / Kamala / Biden / Trump
- Dropdown 2: action verbs (fun, not serious — e.g. "accidentally microwaved", "challenged to a dance-off with", "was caught hoarding")
- Dropdown 3: objects (e.g. "a sentient Roomba", "the Pentagon's WiFi password", "a surplus of Costco mozzarella sticks")
- Generate button → loading state → two articles rendered side by side in styled "newspaper" panes

**Q&A mode** (`QAChat.tsx`):
- Text input with submit
- Two answer panes side by side, each labeled with source badge

---

## Phase 6 — Deployment

- API: RunPod serverless endpoint (keeps GPU warm on first hit) or Modal
- Frontend: Vercel
- Models: stored in Hugging Face Hub (private repo) or RunPod volume

---

## Dropdowns Content (Mad Libs)

**Politicians**: Obama, Hillary, Kamala, Biden, Trump

**Verbs** (compile 15–20 options, e.g.):
- accidentally microwaved
- lost a staring contest with
- challenged to a dance-off with
- was caught hoarding
- issued an executive order about
- offered a high-five to
- made eye contact with
- tried to negotiate with
- was photographed napping near

**Objects** (compile 20–30 options, e.g.):
- a surplus of Costco mozzarella sticks
- the Pentagon's WiFi password
- a sentient Roomba
- three raccoons in a trench coat
- an expired Bed Bath & Beyond coupon
- the last known physical copy of the Yellow Pages

---

## Verification

1. **Scraper**: Run on a small sample (50 articles per source), verify JSONL output schema
2. **Dataset prep**: Check instruction format, token length distribution, train/eval split counts
3. **Training**: Monitor loss curves; eval perplexity should diverge between CNN and Fox models
4. **Bias check**: Prompt both models with neutral topic (e.g. "Trump signs trade bill") — qualitatively verify stylistic/tonal differences
5. **API**: `curl` both endpoints, verify parallel response structure
6. **Frontend**: Test all 5×N×M Mad Libs combos don't break; test Q&A with political and neutral questions
