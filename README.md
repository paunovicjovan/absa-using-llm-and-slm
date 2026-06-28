# Aspect-Based Sentiment Analysis with small and large language models

This repository contains a seminar paper and the accompanying Jupyter notebooks developed for the
course Natural Language Processing, as part of the MSc program in Artificial Intelligence and Machine
Learning at the University of Niš, Faculty of Electronic Engineering.

The project compares small and large language models on aspect-based sentiment analysis (ABSA),
using the SemEval-2014 Task 4 restaurant reviews. The same task is run under three regimes – zero-shot,
few-shot, and fine-tuning – through a single uniform inference interface, so the numbers are
actually comparable across models.

The goal was to see how far small models that run locally on a laptop can be pushed, and how they
hold up against a hosted large model on the same test set.

## The task

Given a single sentence from a restaurant review, the model has to do two things at once:

1. **Aspect Term Extraction (ATE)** – pull out every aspect term mentioned in the sentence, copied
   verbatim (for example `food`, `staff`, `Thai cuisine`).
2. **Aspect Sentiment Classification (ASC)** – assign each extracted aspect one of four sentiments:
   `positive`, `negative`, `neutral`, or `conflict`.

A sentence can contain several aspects with different sentiments, and some sentences contain none at
all. Every model is asked to answer with a plain JSON array, which is then parsed and scored.

## Models

| Model | Type | How it runs |
|-------|------|-------------|
| Gemma 4 E2B | small (local) | Ollama |
| Qwen3.5 2B (q4_K_M) | small (local) | Ollama |
| Gemini 3.5 Flash | large (hosted) | Google GenAI API |

The two small models are also fine-tuned with LoRA / QLoRA and evaluated again, so each one appears in
zero-shot, few-shot, and fine-tuned form.

## Dataset

The source file is the [SemEval-2014 Task 4](https://www.kaggle.com/datasets/charitarth/semeval-2014-task-4-aspectbasedsentimentanalysis?select=Restaurants_Train_v2.csv) restaurant training set (`Restaurants_Train_v2.csv`), which
is in long format – one row per aspect term. The notebook reconstructs it back to sentence level, where
each example is a sentence plus the list of its `(aspect, sentiment)` pairs. Sentences with no aspect
term are kept as empty lists, which matters because a good model should also learn to stay quiet when
there is nothing to extract.

The data is then split once at the sentence level with a fixed seed (1616 train / 405 test). That same
test set is reused by every model in the project, including the fine-tuned ones trained later, so no
model has an easier or harder evaluation than another.

## How the evaluation works

The pipeline is deliberately split into inference and scoring so the two never get tangled:

- Each model exposes the same `predict(prompt) -> (raw_text, seconds)` method, whether it is a local
  Ollama model or the hosted Gemini API. The inference loop never needs to know which kind of model it
  is talking to.
- Predictions are written to JSONL one line at a time. Each record carries the gold pairs alongside the
  model's output, so scoring is self-contained from the file. If a run is interrupted it resumes from
  the last completed sentence instead of starting over.
- Generative models return free text, so parsing happens in two passes: a direct JSON parse, then a
  repair attempt that strips reasoning traces and code fences before extracting the array. If both fail,
  the output is recorded as a format failure rather than silently dropped.
- Metrics are computed afterwards from the stored files, so the scoring section can be re-run and edited
  without ever touching the models again.

### Metrics

- **ATE** – precision / recall / F1 over the set of extracted aspect terms, exact match.
- **ASC** – sentiment accuracy and macro-F1, computed on the aspects the model extracted correctly.
  Macro-F1 is reported because `neutral` and `conflict` are rare, and accuracy alone would hide how
  badly a model does on them.
- **End-to-end F1** – F1 over full `(aspect, sentiment)` pairs, where a pair only counts as correct if
  both the term and its sentiment are right. This is the headline number for comparing models.
- **Compliance** – the share of outputs that parsed at all. Quality metrics are computed only on parsed
  outputs, so compliance always has to be read next to them.

## Results

End-to-end F1 on the 405-sentence test set:

| Model | Zero-shot | Few-shot | Fine-tuned |
|-------|:---------:|:--------:|:----------:|
| Gemini 3.5 Flash | 0.693 | 0.728 | – |
| Gemma 4 E2B | 0.551 | 0.623 | 0.649 |
| Qwen3.5 2B | 0.440 | 0.444 | **0.731** |

A few things stand out:

- Few-shot prompting helps every model over its zero-shot baseline, without any training.
- Fine-tuning a 2B model on roughly 1600 examples is enough to match, and slightly beat, a hosted large
  model on this narrow task – the fine-tuned Qwen3.5 reaches the top end-to-end F1.
- Fine-tuning helped Qwen far more than Gemma, which was already a stronger prompted model to begin with.

Full per-model numbers, including ATE, ASC, compliance, latency, and memory, are in
[`Projekat/results/results.csv`](Projekat/results/results.csv).

## Repository layout

```
Projekat/
  Project.ipynb            main pipeline: data prep, inference, evaluation, plots
  Finetune_Gemma4.ipynb    LoRA fine-tuning + evaluation of Gemma 4 E2B (Colab)
  Finetune_Qwen3_5.ipynb   QLoRA fine-tuning + evaluation of Qwen3.5 2B (Colab)
  data/                    source CSV and the fixed train/test split (json + csv)
  predictions/             raw per-model outputs as JSONL
  results/                 aggregated metrics table
Seminarski rad/            seminar paper (Serbian), .docx and .pdf
```

The fine-tuning notebooks reuse the exact same prompt the main notebook serves at inference time, so
the adapter is trained on precisely the input it later has to answer.

## Running it

The main notebook needs:

- [Ollama](https://ollama.com/) running locally, with the two small models pulled.
- A Gemini API key in `Projekat/.env` as `GEMINI_API_KEY`.
- Python with `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `ollama`, `google-genai`,
  `python-dotenv`, and `adjustText`.

Pull the local models:

```bash
ollama pull gemma4:e2b
ollama pull qwen3.5:2b-q4_K_M
```

Then run `Projekat/Project.ipynb` top to bottom. Because predictions are cached to disk, re-running the
notebook skips work that is already done and only fills in what is missing.

The two fine-tuning notebooks are written for Google Colab (they use Unsloth and a T4 GPU) and write
their predictions into the same `predictions/` folder so they fold straight into the evaluation.
