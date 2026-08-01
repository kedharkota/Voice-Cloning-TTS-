# TeluguVoiceBridge v2 — Complete Project Documentation

## End-to-End Telugu-to-English Speech-to-Speech Translation System

**Hardware:** NVIDIA RTX 4060 Laptop (7.6 GB VRAM) | 24 GB RAM | ~60 GB Storage  
**Architecture:** 5-Model Cascade (Whisper + ECAPA-TDNN + NLLB + Wav2Vec2 + XTTS-v2)  
**Goal:** Convert Telugu speech input into English speech output with preserved speaker voice and emotional tone

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [High-Level System Architecture](#2-high-level-system-architecture)
3. [Pipeline Workflow (End-to-End)](#3-pipeline-workflow-end-to-end)
4. [Notebook-by-Notebook Breakdown](#4-notebook-by-notebook-breakdown)
   - [01 — Environment Setup](#41-notebook-01--environment-setup--system-checks)
   - [02 — Data Pipeline](#42-notebook-02--data-pipeline)
   - [03 — Whisper ASR Fine-Tuning](#43-notebook-03--whisper-asr-fine-tuning)
   - [04 — Speaker Encoder Fine-Tuning](#44-notebook-04--speaker-encoder-fine-tuning-ecapa-tdnn)
   - [05 — Translation Fine-Tuning](#45-notebook-05--translation-fine-tuning-indictrans2--nllb)
   - [06 — Emotion Detector](#46-notebook-06--emotion-detector-wav2vec2--vad)
   - [07 — TTS Voice Cloning](#47-notebook-07--tts-voice-cloning-xtts-v2)
   - [07b — TTS Rebuild](#48-notebook-07b--tts-rebuild-3-tier-fix)
   - [08 — Pipeline Integration & Demo](#49-notebook-08--pipeline-integration--demo)
5. [Model Architectures In-Depth](#5-model-architectures-in-depth)
6. [Training Methodology & Optimization Techniques](#6-training-methodology--optimization-techniques)
7. [Data Pipeline Architecture](#7-data-pipeline-architecture)
8. [VRAM & Storage Budget](#8-vram--storage-budget)
9. [Evaluation Framework](#9-evaluation-framework)
10. [Ablation Studies](#10-ablation-studies)
11. [Key Technical Innovations & Bug Fixes](#11-key-technical-innovations--bug-fixes)
12. [Python Scripts](#12-python-scripts)
13. [File & Directory Structure](#13-file--directory-structure)
14. [Target Metrics vs Achieved Results](#14-target-metrics-vs-achieved-results)
15. [Future Enhancements](#15-future-enhancements)
16. [References](#16-references)

---

## 1. Project Overview

### What is TeluguVoiceBridge v2?

TeluguVoiceBridge v2 is an **end-to-end speech-to-speech translation system** that takes Telugu audio as input and produces English audio as output. Crucially, the output English speech **preserves the original speaker's voice identity and emotional tone** — making it sound like the same person is now speaking English.

### Why is this hard?

Speech-to-speech translation is a multi-faceted problem that requires solving 5 sub-problems simultaneously:

| Sub-Problem | What it does | Model Used |
|---|---|---|
| **Automatic Speech Recognition (ASR)** | Converts Telugu audio → Telugu text | Whisper-large-v3 (INT8 + LoRA) |
| **Speaker Identity Extraction** | Captures "who is speaking" as a 192-dim vector | ECAPA-TDNN (SpeechBrain) |
| **Machine Translation (MT)** | Translates Telugu text → English text | NLLB-200-distilled-600M (4-bit QLoRA) |
| **Emotion Detection** | Extracts emotional state as Valence/Arousal/Dominance | Wav2Vec2-base (regression head) |
| **Text-to-Speech (TTS)** | Synthesizes English speech using cloned voice + emotion | XTTS-v2 (Coqui TTS) |

### Core Design Constraints

- **VRAM Limit:** Everything must run on an 8 GB GPU (7.6 GB usable)
- **No Cloud:** All training and inference happen locally on a single RTX 4060
- **Sequential Training:** Models are trained one at a time, then all loaded together for inference
- **Parallel Inference:** Three branches (ASR→MT, Speaker, Emotion) run in parallel to minimize latency

---

## 2. High-Level System Architecture

### 2.1 Complete Pipeline Diagram

```
                            ┌──────────────────────────────────────────┐
                            │         INPUT: Telugu Speech (WAV)       │
                            └─────────────────┬────────────────────────┘
                                              │
                          ┌───────────────────┼───────────────────┐
                          │                   │                   │
                          ▼                   ▼                   ▼
               ┌──────────────────┐ ┌─────────────────┐ ┌────────────────────┐
               │  BRANCH A (GPU)  │ │  BRANCH B (CPU) │ │  BRANCH C (GPU)    │
               │                  │ │                 │ │                    │
               │  Whisper ASR     │ │  ECAPA-TDNN     │ │  Wav2Vec2          │
               │  (INT8 + LoRA)   │ │  Speaker Enc.   │ │  Emotion Detector  │
               │                  │ │                 │ │                    │
               │  Telugu Audio    │ │  Telugu Audio   │ │  Telugu Audio      │
               │      ↓           │ │      ↓          │ │      ↓             │
               │  Telugu Text     │ │  192-dim        │ │  [V, A, D]         │
               │      ↓           │ │  d-vector       │ │  emotion vector    │
               │  IndicTrans2/    │ │  (speaker ID)   │ │                    │
               │  NLLB (4-bit)    │ │                 │ │                    │
               │      ↓           │ │                 │ │                    │
               │  English Text    │ │                 │ │                    │
               └────────┬─────────┘ └────────┬────────┘ └─────────┬──────────┘
                        │                    │                    │
                        └────────────────────┼────────────────────┘
                                             │
                                             ▼
                              ┌──────────────────────────────┐
                              │         TTS Engine           │
                              │         XTTS-v2              │
                              │                              │
                              │  Inputs:                     │
                              │   • English text             │
                              │   • Speaker d-vector         │
                              │   • VAD emotion vector       │
                              │   • Reference audio          │
                              │                              │
                              │  Output:                     │
                              │   • English speech (24kHz)   │
                              │   • In original speaker's    │
                              │     voice + emotion          │
                              └──────────────┬───────────────┘
                                             │
                                             ▼
                              ┌──────────────────────────────┐
                              │  OUTPUT: English Speech (WAV)│
                              │  Same voice, same emotion    │
                              └──────────────────────────────┘
```

### 2.2 Device Allocation Strategy

| Component | Device | VRAM/RAM | Reason |
|---|---|---|---|
| Whisper ASR (INT8) | GPU | ~2.5 GB | Real-time transcription needs GPU speed |
| ECAPA-TDNN | **CPU** | ~0.3 GB RAM | Only 80 MB model — CPU is fast enough, frees GPU |
| NLLB Translation (4-bit) | GPU | ~0.8 GB | Quantized small enough to coexist with Whisper |
| Emotion Detector | GPU | ~0.4 GB | Small head on Wav2Vec2 base |
| XTTS-v2 TTS | GPU | ~0.8 GB | Speech synthesis is compute-intensive |
| CUDA Overhead | GPU | ~1.0 GB | PyTorch allocator, kernels, buffers |
| **Total GPU** | | **~5.1 GB** | **Fits in 8 GB with 2.9 GB headroom** |

---

## 3. Pipeline Workflow (End-to-End)

### Step-by-Step Data Flow

```
STEP 1: Load Audio
    ├── torchaudio.load(audio_path)
    ├── Resample to 16 kHz
    └── Convert to mono (if stereo)

STEP 2: Parallel Processing (3 branches)
    ├── BRANCH A [GPU, sequential]:
    │   ├── Whisper ASR: audio → Telugu text
    │   │   ├── Feature extraction (80-dim mel spectrogram)
    │   │   ├── Encoder (12 transformer blocks, INT8)
    │   │   ├── Decoder with forced language="te", task="transcribe"
    │   │   └── Output: Telugu text (Devanagari/Telugu script)
    │   │
    │   └── NLLB Translation: Telugu text → English text
    │       ├── Tokenize with src_lang="tel_Telu"
    │       ├── Encoder-Decoder (4-bit NF4 + LoRA)
    │       ├── Beam search (num_beams=4)
    │       ├── forced_bos_token_id = eng_Latn
    │       └── Output: English text
    │
    ├── BRANCH B [CPU, threaded]:
    │   └── ECAPA-TDNN Speaker Encoder
    │       ├── encode_batch(audio_tensor)
    │       ├── L2 normalize
    │       └── Output: 192-dim d-vector (speaker identity)
    │
    └── BRANCH C [GPU]:
        └── Wav2Vec2 Emotion Detector
            ├── Feature extraction (frozen)
            ├── 6 trainable transformer layers
            ├── Regression head (768→128→3, Tanh)
            └── Output: [Valence, Arousal, Dominance] ∈ [-1, 1]

STEP 3: Join All Branches
    ├── Wait for CPU thread to complete
    └── Collect: english_text, d-vector, vad_vector

STEP 4: TTS Synthesis
    ├── XTTS-v2 voice cloning
    ├── Input: english_text + speaker_wav (original audio as reference)
    ├── FiLM conditioning with VAD emotion vector
    └── Output: English audio at 24 kHz

STEP 5: Return Results
    ├── Telugu transcript
    ├── English translation
    ├── English audio (WAV)
    ├── Speaker embedding
    ├── VAD vector
    └── Per-stage timing breakdown
```

### Long Audio Handling

For audio longer than ~30 seconds, the pipeline uses **WebRTC VAD (Voice Activity Detection)** to split audio into speech segments:

```
Long Audio → WebRTC VAD (aggressiveness=2, 30ms frames)
    → List of speech segments [(start, end), ...]
    → Process each segment through full pipeline
    → Crossfade concat (50ms linear crossfade)
    → Merged English audio output
```

---

## 4. Notebook-by-Notebook Breakdown

### 4.1 Notebook 01 — Environment Setup & System Checks

**Purpose:** Bootstrap the entire project environment

**What it does (9 tasks):**

| Task | Description |
|---|---|
| **System Health Checks** | Validates GPU (nvidia-smi), RAM (≥16 GB), disk (≥20 GB free), swap |
| **Directory Structure** | Creates 22 folders for data, checkpoints, configs, results, logs |
| **Dependency Installation** | Installs packages in order: PyTorch → transformers/peft/accelerate → bitsandbytes → datasets/audio libs → SpeechBrain/Coqui TTS → evaluation tools → UTMOS |
| **Import Verification** | Programmatically imports all 18 required packages and reports status |
| **YAML Config Generation** | Writes 5 config files: `asr.yaml`, `speaker.yaml`, `translation.yaml`, `emotion.yaml`, `tts.yaml` |
| **VRAM Budget Doc** | Documents memory usage per training phase (all fit within 7.6 GB) |
| **Swap Setup** | Optional 16 GB swap file creation instructions |
| **GPU Smoke Test** | Runs FP16 forward+backward pass on CUDA to verify GPU works |
| **Storage Budget** | Documents the 60 GB disk allocation |

**Config files generated:**

| Config | Key Settings |
|---|---|
| `asr.yaml` | Whisper-large-v3, INT8, LoRA r=8/α=16, SpecAugment params, target WER ≤ 20% |
| `speaker.yaml` | ECAPA-TDNN, CPU training, AAM-Softmax (margin=0.2, scale=30), target EER ≤ 12% |
| `translation.yaml` | IndicTrans2-200M / NLLB fallback, 4-bit QLoRA, tel_Telu → eng_Latn, target BLEU ≥ 22 |
| `emotion.yaml` | Wav2Vec2-base, freeze 6 layers, VAD output dim=3, MSE+CCC loss, target CCC ≥ 0.55 |
| `tts.yaml` | XTTS-v2, Phase 5a (LJSpeech+VCTK 50k steps), Phase 5b (cross-lingual 20k steps), target SECS ≥ 0.70 |

---

### 4.2 Notebook 02 — Data Pipeline

**Purpose:** Download, preprocess, and build metadata manifests for all 4 datasets

**Datasets processed:**

| # | Dataset | Purpose | Raw Size | Processed Size |
|---|---|---|---|---|
| 1 | **CoVoST-2 Telugu** | ASR + Translation training | ~3 GB | ~2.5 GB |
| 2 | **AI4Bharat Rasa Telugu** | Speaker Encoder + ASR | ~2 GB | ~1.5 GB |
| 3 | **VCTK** | TTS multi-speaker training | ~11 GB | ~8 GB |
| 4 | **LJSpeech** | TTS baseline training | ~3 GB | ~2.5 GB |

**Critical Memory Management Rule:**
> Download ONE dataset → process → delete raw → download next.  
> Never have more than one raw dataset on disk simultaneously.

**Processing pipeline for each dataset:**

```
Raw Audio → Resample to 16 kHz → Mono → Loudness Normalize (-1 dB peak)
    → Duration Filter (0.5s–30s) → Quality Check → Save as 16-bit PCM WAV
    → Build CSV manifest (path, text, duration, speaker_id, split)
```

**Utility functions defined:**
- `normalize_audio(wav, sr)` — resample + mono + peak normalize to 0.9
- `save_wav(wav, path)` — write 16-bit PCM WAV
- `check_audio(path)` — verify file readability, return duration

**Output manifests:**
- `data/metadata/asr_manifest.csv` — ASR training data (audio path + Telugu text)
- `data/metadata/speaker_manifest.csv` — Speaker encoder training data
- `data/metadata/tts_manifest.csv` — TTS training data
- `data/metadata/emotion_manifest.csv` — Emotion detector training data
- `data/metadata/translation_pairs.csv` — Telugu→English text pairs for MT

**Data splits:** Train / Validation / Test (80/10/10 or dataset-specific)

---

### 4.3 Notebook 03 — Whisper ASR Fine-Tuning

**Purpose:** Fine-tune Whisper-large-v3 for Telugu speech recognition

**Architecture:**

```
Telugu Audio (16kHz)
    ↓
Mel Spectrogram (80 bins × 3000 frames)
    ↓
┌─────────────────────────────────────────┐
│  Whisper Encoder (12 transformer blocks) │
│  INT8 Quantized + LoRA (r=8, α=16)      │
│  8 attention heads, dim=768              │
└───────────────────┬─────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│  Whisper Decoder (12 transformer blocks) │
│  INT8 Quantized + LoRA                   │
│  forced language="te", task="transcribe" │
└───────────────────┬─────────────────────┘
                    ↓
Telugu Text Output (Devanagari/Telugu script)
```

**Model & Quantization:**
- **Base model:** `openai/whisper-large-v3` (~2.4 GB → ~2.5 GB in INT8)
- **Quantization:** INT8 via `BitsAndBytesConfig(load_in_8bit=True)`
- **LoRA:** r=8, α=16, targets `q_proj` + `v_proj`, no `task_type` (avoids PEFT bug)

**Data Augmentation (3 on-the-fly transforms):**

| Technique | Details | Purpose |
|---|---|---|
| **Speed Perturbation** | Random resample at 0.9×/1.0×/1.1× | Robustness to speaking rate |
| **Additive Noise** | Gaussian noise at configurable SNR | Handling noisy environments |
| **SpecAugment** | 2 time masks (max 50 frames) + 2 freq masks (max 20 bins), 80% prob | Regularization on mel features |

**Training Configuration:**
- **Training loop:** Custom plain PyTorch loop (NOT HuggingFace Trainer — avoids PEFT+Whisper compatibility bugs)
- **Batch size:** 1 (VRAM limited), gradient accumulation = 8 → effective batch = 8
- **Optimizer:** 8-bit AdamW (`bitsandbytes.optim.AdamW8bit`)
- **LR Schedule:** Cosine with linear warmup
- **Mixed precision:** FP16 via `torch.amp.GradScaler`
- **Gradient clipping:** Norm ≤ 1.0
- **Checkpointing:** Every 250 steps, best model by WER, max 2 saved

**Evaluation:**
- **Metrics:** WER (Word Error Rate) and CER (Character Error Rate) via `jiwer`
- **Target:** WER ≤ 20%, CER ≤ 10%

**Post-training:**
- Best LoRA adapter merged into base model via `merge_and_unload()`
- Merged model exported to `checkpoints/whisper_merged/` for production inference

---

### 4.4 Notebook 04 — Speaker Encoder Fine-Tuning (ECAPA-TDNN)

**Purpose:** Fine-tune a speaker encoder to produce language-invariant speaker embeddings

**Why this matters:** Without this, the speaker embedding would encode both voice identity AND language characteristics. During Telugu→English TTS synthesis, this causes voice mismatch. The Gradient Reversal Layer (GRL) strips language information, preserving only speaker timbre.

**Architecture:**

```
Audio Input (16kHz, max 8s)
    ↓
┌─────────────────────────────────────────┐
│  ECAPA-TDNN Encoder                      │
│  (SpeechBrain spkrec-ecapa-voxceleb)     │
│  ~80 MB, trained on VoxCeleb2            │
└───────────────┬─────────────────────────┘
                ↓
    192-dim embedding (L2-normalized)
                ↓
        ┌───────┴───────┐
        ↓               ↓
┌──────────────┐ ┌──────────────────┐
│ Speaker Head │ │ Language Head     │
│ AAM-Softmax  │ │ (GRL + Classifier)│
│ (margin=0.2, │ │ en=0, te=1        │
│  scale=30)   │ │ λ: 0→1 annealing │
└──────────────┘ └──────────────────┘
```

**Key Design Decisions:**

| Decision | Why |
|---|---|
| **CPU Training** | Model is only 80 MB — CPU is feasible (~4-6 hrs/epoch), frees GPU for Whisper |
| **AAM-Softmax Loss** | Angular margin penalty $\cos(\theta + m)$ enforces tight intra-speaker clustering |
| **Gradient Reversal Layer** | Adversarial branch forces language-invariant embeddings |
| **GRL λ Annealing** | λ ramps from 0→1 over first half of training (Ganin et al. 2016) |
| **Contrastive Batching** | 4 speakers × 4 utterances per batch for diverse representation |

**Training Data:**
- RAVDESS (English speakers) for speaker labels
- FLEURS Telugu for adversarial language branch

**Evaluation Metrics:**

| Metric | Target | Method |
|---|---|---|
| **EER** (Equal Error Rate) | ≤ 12% | Cosine similarity on same/different pairs → ROC → Brent's method |
| **Cosine Similarity** (same-speaker) | ≥ 0.70 | Mean cosine sim among same-speaker pairs |
| **Inference Latency** | ≤ 0.5s | 10-run average for 5s clip → 192-dim embedding on CPU |

**Output:** 192-dim L2-normalized d-vector (speaker identity fingerprint)

---

### 4.5 Notebook 05 — Translation Fine-Tuning (IndicTrans2 / NLLB)

**Purpose:** Fine-tune a multilingual translation model for Telugu→English

**Architecture:**

```
Telugu Text (tel_Telu tokens)
    ↓
┌─────────────────────────────────────────┐
│  NLLB Encoder (12 transformer blocks)    │
│  4-bit NF4 Quantized + LoRA (r=8, α=16) │
│  16 attention heads, dim=1024            │
└───────────────────┬─────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│  NLLB Decoder (12 transformer blocks)    │
│  4-bit NF4 Quantized + LoRA             │
│  forced_bos_token_id = eng_Latn          │
│  num_beams=4 (beam search)               │
└───────────────────┬─────────────────────┘
                    ↓
English Text (eng_Latn tokens)
```

**Model & Quantization:**
- **Primary:** `ai4bharat/indictrans2-indic-en-dist-200M` (gated, may not be accessible)
- **Fallback:** `facebook/nllb-200-distilled-600M` (600M params, 200 languages)
- **Quantization:** 4-bit NF4 via QLoRA (`load_in_4bit=True`, double quantization, FP16 compute)

**Tokenization (Critical Detail):**
```python
# Source: tokenize normally after setting src_lang
tokenizer.src_lang = "tel_Telu"
source_tokens = tokenizer(telugu_text, ...)

# Target: use text_target= keyword (NOT as_target_tokenizer() context manager!)
target_tokens = tokenizer(text_target=english_text, ...)

# Labels: replace padding with -100 (ignored in loss)
labels[labels == tokenizer.pad_token_id] = -100
```

**Training:**
- Custom plain PyTorch loop
- 8-bit AdamW optimizer
- OneCycleLR scheduler (linear warmup + cosine decay)
- FP16 mixed precision + GradScaler
- Gradient clipping at 1.0

**Evaluation:**
- **BLEU** via `sacrebleu.corpus_bleu` (target: ≥ 22 MT-only, ≥ 15 pipeline)
- Beam search with `num_beams=4`, `max_new_tokens=128`

**Post-processing:**
- Telugu input: NFC Unicode normalization + repeated char dedup + whitespace collapse
- English output: Capitalize first letter + ensure trailing period

---

### 4.6 Notebook 06 — Emotion Detector (Wav2Vec2 → VAD)

**Purpose:** Detect continuous emotional state from speech audio as 3D Valence-Arousal-Dominance vector

**Why continuous VAD instead of discrete labels?**
- Discrete labels ("happy", "sad") lose **intensity** — "slightly irritated" ≠ "furious"
- VAD provides **smooth conditioning signal** for TTS FiLM layers
- Enables **interpolation** between emotional states at inference time

**Architecture:**

```
Audio Input (16kHz)
    ↓
┌─────────────────────────────────────────┐
│  Wav2Vec2-base Feature Extractor         │
│  (FROZEN — no gradients)                 │
└───────────────────┬─────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│  Wav2Vec2 Transformer Encoder            │
│  Layers 0-5: FROZEN                      │
│  Layers 6-11: TRAINABLE                  │
└───────────────────┬─────────────────────┘
                    ↓
    Mean Pooling (over time dimension)
                    ↓
┌─────────────────────────────────────────┐
│  VAD Regression Head                     │
│  Dropout(0.1)                            │
│  → Linear(768 → 128) → ReLU             │
│  → Dropout(0.1)                          │
│  → Linear(128 → 3) → Tanh               │
│  Output: [Valence, Arousal, Dominance]   │
│  Each ∈ [-1, 1]                          │
└─────────────────────────────────────────┘
```

**RAVDESS Emotion → VAD Mapping (Russell & Mehrabian):**

| Emotion | Valence | Arousal | Dominance |
|---|---|---|---|
| Neutral | 0.0 | 0.0 | 0.0 |
| Calm | +0.3 | -0.3 | +0.1 |
| Happy | +0.8 | +0.5 | +0.3 |
| Sad | -0.7 | -0.3 | -0.4 |
| Angry | -0.5 | +0.7 | +0.6 |
| Fearful | -0.6 | +0.6 | -0.4 |
| Disgust | -0.6 | +0.2 | +0.3 |
| Surprised | +0.3 | +0.7 | -0.1 |

**Loss Function — MSE + CCC:**

$$\mathcal{L} = \text{MSE}(y, \hat{y}) + 0.5 \times (1 - \overline{\text{CCC}})$$

where CCC (Concordance Correlation Coefficient) per dimension:

$$\text{CCC} = \frac{2 \sigma_{xy}}{\sigma_x^2 + \sigma_y^2 + (\mu_x - \mu_y)^2}$$

**Data Augmentation:**
- Speed perturbation (0.9–1.1×)
- Pitch shift (±2 semitones)
- Additive Gaussian noise (~40 dB SNR)

**Target:** Mean CCC ≥ 0.55

---

### 4.7 Notebook 07 — TTS Voice Cloning (XTTS-v2)

**Purpose:** Generate English speech that sounds like the original Telugu speaker

**Approach:** Zero-shot voice cloning using pretrained XTTS-v2 multilingual model

**Why no fine-tuning?**
1. XTTS-v2 requires Coqui's internal `Trainer` framework (not plain PyTorch)
2. Fine-tuning needs 50k+ steps (~15h) — impractical on constrained hardware
3. Pretrained model already provides good voice cloning quality

**FiLM Conditioning (Feature-wise Linear Modulation):**

```
VAD Vector [V, A, D]
    ↓
MLP: Linear(3→hidden) → ReLU → split into γ and β
    ↓
output = γ(vad) · hidden_features + β(vad)
```

FiLM layers are inserted after each HiFi-GAN decoder upsampling stage (4 layers total). The VAD emotion vector modulates the speech features to inject emotional tone.

> **Note:** FiLM layers are **defined** in NB07 but **integrated** in NB08.

**Speaker Embedding for Voice Cloning:**
- Reference WAV passed as `speaker_wav` parameter
- SECS (Speaker Embedding Cosine Similarity) evaluated using ECAPA-TDNN
- Target: SECS ≥ 0.70

---

### 4.8 Notebook 07b — TTS Rebuild (3-Tier Fix)

**Problem:** NB07's XTTS-v2 output was garbled ("grr grr gha gha")

**Root Causes Identified:**
1. `repetition_penalty=10.0` (5× the config default) → audio loops/garbles
2. `temperature=0.75` too high → random token sampling
3. 3.3s RAVDESS acted speech as reference → bad voice conditioning
4. High-level `tts_api.tts()` hides critical generation parameters

**3-Tier Fix Strategy:**

| Tier | Approach | Details |
|---|---|---|
| **Tier 1** | Fix XTTS-v2 | Low-level `model.inference()` API, `repetition_penalty=2.0`, `temperature=0.3`, LJSpeech 8-10s refs, `gpt_cond_len=12` |
| **Tier 2** | F5-TTS | Flow-matching diffusion transformer (~330M params), non-autoregressive (no garbling by design), ~4 GB VRAM |
| **Tier 3** | YourTTS | VITS-based (end-to-end, no autoregressive), multi-speaker/multi-lingual, ~2× faster than XTTS |

**Selection:** A/B comparison across all 3 tiers on duration ratio, RMS energy, inference speed, and subjective quality determines the final engine for the pipeline.

---

### 4.9 Notebook 08 — Pipeline Integration & Demo

**Purpose:** Load all 5 models simultaneously, wire them into the full pipeline, evaluate, and demo

**Contents:**

| Section | Description |
|---|---|
| **Model Loading** | Load 5 models sequentially with proper quantization/device placement |
| **Pipeline Function** | `telugu_voice_bridge(audio_path)` — full A/B/C branch pipeline |
| **Long Audio** | WebRTC VAD chunking + 50ms crossfade concatenation |
| **Evaluation Suite** | WER, CER, BLEU, chrF, SECS, CCC metrics on test split |
| **Ablation Studies** | 3 ablations: no FiLM, no contrastive fine-tuning, discrete vs continuous VAD |
| **Batch Inference** | 10 sample I/O pairs with JSON logging |
| **Gradio Demo** | Web UI with audio upload/mic recording, shows transcript + translation + audio |

**Gradio Demo:**
- Single-threaded (`max_threads=1`) to stay within 8 GB VRAM
- Audio input (file upload or microphone)
- 5 outputs: English audio, Telugu transcript, English translation, SECS score, latency breakdown

---

## 5. Model Architectures In-Depth

### 5.1 Whisper-large-v3 (ASR)

```
┌──────────────────────────────────────────────────────────┐
│                    WHISPER-LARGE-V3                       │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ENCODER:                                                │
│  ├── Conv1D (2 layers) — 80-dim mel → 1280-dim           │
│  ├── Sinusoidal positional embeddings                    │
│  └── 32 Transformer blocks                               │
│       ├── Multi-head self-attention (20 heads, dim=1280) │
│       ├── Feed-forward (5120 hidden)                     │
│       └── Layer norm (pre-norm)                          │
│                                                          │
│  DECODER:                                                │
│  ├── Learned positional embeddings                       │
│  └── 32 Transformer blocks                               │
│       ├── Masked self-attention (20 heads)               │
│       ├── Cross-attention to encoder (20 heads)          │
│       ├── Feed-forward (5120 hidden)                     │
│       └── Layer norm (pre-norm)                          │
│                                                          │
│  Parameters: ~1.5B total → ~12M trainable via LoRA      │
│  Input: 80-dim × 3000-frame mel spectrogram              │
│  Output: token sequence (max 225 tokens)                 │
│  Vocabulary: Multilingual (includes Telugu)              │
│  Quantization: INT8 (BitsAndBytes)                       │
│  LoRA: r=8, α=16, targets q_proj + v_proj               │
└──────────────────────────────────────────────────────────┘
```

### 5.2 ECAPA-TDNN (Speaker Encoder)

```
┌──────────────────────────────────────────────────────────┐
│                    ECAPA-TDNN                             │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  1D Conv layers with Squeeze-Excitation (SE) blocks      │
│  Multi-layer feature aggregation (MFA)                   │
│  Channel and context dependent statistics pooling        │
│  Attentive statistical pooling (ASP)                     │
│                                                          │
│  Input: Raw waveform (16kHz, up to 8s)                   │
│  Output: 192-dim L2-normalized embedding (d-vector)      │
│  Size: ~80 MB                                            │
│  Pre-trained on: VoxCeleb2 (6000+ speakers)              │
│  Fine-tuned with: AAM-Softmax + GRL                     │
│  Device: CPU                                             │
└──────────────────────────────────────────────────────────┘
```

### 5.3 NLLB-200-distilled-600M (Translation)

```
┌──────────────────────────────────────────────────────────┐
│                    NLLB-200-distilled-600M                │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ENCODER:                                                │
│  └── 12 Transformer blocks                               │
│       ├── Multi-head self-attention (16 heads, dim=1024) │
│       ├── Feed-forward (4096 hidden)                     │
│       └── Layer normalization                            │
│                                                          │
│  DECODER:                                                │
│  └── 12 Transformer blocks                               │
│       ├── Masked self-attention (16 heads)               │
│       ├── Cross-attention to encoder (16 heads)          │
│       ├── Feed-forward (4096 hidden)                     │
│       └── Layer normalization                            │
│                                                          │
│  Parameters: 600M total, ~few M trainable via LoRA      │
│  Languages: 200 (including tel_Telu ↔ eng_Latn)         │
│  Vocabulary: 256K tokens (multilingual)                  │
│  Quantization: 4-bit NF4 (QLoRA)                        │
│  LoRA: r=8, α=16, targets q_proj + v_proj               │
└──────────────────────────────────────────────────────────┘
```

### 5.4 Wav2Vec2-base (Emotion Detector)

```
┌──────────────────────────────────────────────────────────┐
│              WAV2VEC2-BASE + VAD HEAD                    │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Feature Extractor (FROZEN):                             │
│  └── 7 Conv layers → 768-dim features                   │
│                                                          │
│  Transformer Encoder:                                    │
│  ├── Layers 0-5: FROZEN                                  │
│  └── Layers 6-11: TRAINABLE                              │
│       ├── Self-attention (12 heads, dim=768)             │
│       └── Feed-forward (3072 hidden)                     │
│                                                          │
│  VAD Regression Head:                                    │
│  ├── Dropout(0.1)                                        │
│  ├── Linear(768 → 128) + ReLU                            │
│  ├── Dropout(0.1)                                        │
│  └── Linear(128 → 3) + Tanh → [V, A, D]                 │
│                                                          │
│  Output: [Valence, Arousal, Dominance] ∈ [-1, 1]³       │
└──────────────────────────────────────────────────────────┘
```

### 5.5 XTTS-v2 (Text-to-Speech)

```
┌──────────────────────────────────────────────────────────┐
│                    XTTS-v2 (Coqui TTS)                   │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Text Encoder:                                           │
│  └── GPT-2 style transformer (autoregressive)            │
│       Language-specific embeddings                       │
│                                                          │
│  Speaker Conditioning:                                   │
│  └── Reference audio → speaker latents                   │
│       get_conditioning_latents(ref_wav)                  │
│                                                          │
│  Audio Decoder:                                          │
│  └── HiFi-GAN vocoder with FiLM conditioning            │
│       4 upsampling stages                                │
│       FiLM layer after each stage (VAD → γ, β)          │
│                                                          │
│  Input: text + speaker_wav + language code               │
│  Output: waveform at 24 kHz                              │
│  Languages: 16 (including English, Telugu)               │
│  Voice Cloning: Zero-shot from 3-15s reference           │
│  VRAM: ~0.8 GB (inference)                               │
└──────────────────────────────────────────────────────────┘
```

---

## 6. Training Methodology & Optimization Techniques

### 6.1 Training Schedule

All models are trained **sequentially** (one at a time) except the Speaker Encoder (CPU) which can run in parallel with GPU training:

```
Phase 1: Whisper ASR Training          [GPU, ~5-6 hours]
    ↓ save adapter → clear GPU memory
Phase 2: Speaker Encoder Training      [CPU, ~4-6 hours] (can overlap with Phase 1)
    ↓ save checkpoint
Phase 3: NLLB Translation Training     [GPU, ~1-2 hours]
    ↓ save adapter → clear GPU memory
Phase 4: Emotion Detector Training     [GPU, ~2-3 hours]
    ↓ save checkpoint → clear GPU memory
Phase 5: TTS (pretrained, no training) [GPU, only inference]
    ↓
Phase 6: Integration & Evaluation      [GPU+CPU, all models loaded]
```

### 6.2 Memory Optimization Techniques

| Technique | Applied To | What It Does | Memory Saved |
|---|---|---|---|
| **INT8 Quantization** | Whisper | 16-bit → 8-bit weights | ~50% model size |
| **4-bit NF4 Quantization** | NLLB | 16-bit → 4-bit weights | ~75% model size |
| **LoRA** | Whisper, NLLB | Train only 1-2% of parameters | ~95% trainable params |
| **8-bit AdamW** | All GPU models | Optimizer states in 8-bit | ~75% optimizer memory |
| **Gradient Accumulation** | All | Batch=1 × accum=8 → effective=8 | Minimal activation memory |
| **FP16 Mixed Precision** | All GPU | Half-precision activations | ~50% activation memory |
| **CPU Offloading** | Speaker Encoder | Full model on CPU | GPU VRAM freed entirely |
| **Gradient Clipping** | All | Clip norm ≤ 1.0 | Prevents gradient explosion |
| **Sequential Loading** | All phases | One model at a time during training | Never >1 large model on GPU |

### 6.3 LoRA (Low-Rank Adaptation) — How It Works

```
Standard weight update:  W' = W + ΔW              (all params)
LoRA weight update:      W' = W + s · B × A        (low-rank)

Where:
  W: original weight matrix (d × k) — FROZEN
  B: down-projection (d × r) — TRAINABLE
  A: up-projection (r × k) — TRAINABLE
  r: rank (8 in this project, r << min(d, k))
  s: scaling factor = α / r = 16 / 8 = 2

Forward pass:
  output = W(x) + s · B(A(x))
```

**Why LoRA works here:**
- Whisper is pre-trained on 680k hours of multilingual audio → most knowledge is already present
- LoRA adds a small "correction" for Telugu-specific patterns
- Only ~1-2% of parameters need gradients → massive memory savings

### 6.4 Why Custom PyTorch Loops (Not HF Trainer)

The project deliberately avoids HuggingFace's `Trainer` class because of known bugs:

1. **PEFT + Whisper bug:** HF Trainer injects `input_ids` into the encoder (Whisper needs `input_features`)
2. **Accelerate wrapper conflicts:** Multiple wrapper layers (PEFT → Accelerate → DataParallel) cause `decoder_input_ids` errors
3. **Quantization + gradient checkpointing:** BitsAndBytes 4-bit + gradient checkpointing causes gradient flow issues

**Solution:** Manual training loops with explicit control over:
- Model unwrapping (bypass Accelerate wrappers)
- Input preparation (`input_features` for Whisper, proper tokenization for NLLB)
- Gradient flow (disable gradient checkpointing for 4-bit models)
- Memory management (`gc.collect()` + `torch.cuda.empty_cache()` between phases)

---

## 7. Data Pipeline Architecture

### 7.1 Overall Data Flow

```
┌────────────────────────────────────────────────────────────────┐
│                    RAW DATA SOURCES                             │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  CoVoST-2 Telugu       → ASR training + Translation pairs      │
│  AI4Bharat Rasa Telugu → Speaker encoder + Additional ASR      │
│  VCTK                  → TTS multi-speaker                     │
│  LJSpeech              → TTS baseline + reference clips        │
│  RAVDESS               → Emotion detector (8 discrete→VAD)    │
│                                                                │
└────────────────────────────┬───────────────────────────────────┘
                             │ Download one at a time
                             ▼
┌────────────────────────────────────────────────────────────────┐
│                    PREPROCESSING                               │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Audio Processing:              Text Processing:               │
│  • Resample to 16 kHz           • NFC Unicode normalization    │
│  • Convert to mono              • Whitespace collapse          │
│  • Peak normalize to -1 dB      • Repeated char dedup          │
│  • Duration filter (0.5-30s)    • Special char handling        │
│  • Save as 16-bit PCM WAV       • Train/Val/Test split         │
│                                                                │
└────────────────────────────┬───────────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────────┐
│                    MANIFESTS (CSV)                              │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  asr_manifest.csv        → {audio_path, telugu_text,           │
│                              duration, split}                  │
│  speaker_manifest.csv    → {audio_path, speaker_id, split}    │
│  tts_manifest.csv        → {audio_path, text, speaker_id,     │
│                              duration, split}                  │
│  emotion_manifest.csv    → {audio_path, emotion_label,         │
│                              V, A, D, split}                   │
│  translation_pairs.csv   → {telugu_text, english_text, split} │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 7.2 Memory-Safe Processing Pattern

```python
# CRITICAL: Process in batches of 50 to avoid OOM
for batch_start in range(0, total_files, 50):
    batch = files[batch_start:batch_start+50]
    for file in batch:
        process(file)
    gc.collect()  # Force garbage collection after each batch
```

---

## 8. VRAM & Storage Budget

### 8.1 VRAM Budget (Training — One Model at a Time)

```
╔══════════════════════════════════════════════════════════╗
║            VRAM BUDGET — 8 GB GPU (7.6 GB usable)      ║
╠══════════════════════════════════════════════════════════╣
║                                                        ║
║  Whisper ASR Training:                                 ║
║  │ Model (INT8):          2.5 GB                       ║
║  │ LoRA params:           0.1 GB                       ║
║  │ Optimizer (8-bit):     0.4 GB                       ║
║  │ Activations (b=1):    1.5 GB                       ║
║  │ CUDA overhead:         1.0 GB                       ║
║  │ TOTAL:                 5.5 GB  ✓                    ║
║                                                        ║
║  IndicTrans2 Training:                                 ║
║  │ Model (4-bit):        ~1.0 GB                       ║
║  │ Optimizer + grads:    ~2.0 GB                       ║
║  │ CUDA overhead:        ~1.0 GB                       ║
║  │ TOTAL:                ~4.0 GB  ✓                    ║
║                                                        ║
║  XTTS-v2 (Inference Only):                             ║
║  │ Model:                 0.8 GB                       ║
║  │ Activations:           1.5 GB                       ║
║  │ CUDA overhead:         1.0 GB                       ║
║  │ TOTAL:                ~3.3 GB  ✓                    ║
║                                                        ║
║  Speaker Encoder: 0.0 GB GPU (CPU only, ~80 MB RAM)   ║
╚══════════════════════════════════════════════════════════╝
```

### 8.2 VRAM Budget (Inference — All Models Loaded)

```
╔══════════════════════════════════════════════════════════╗
║            INFERENCE VRAM — ALL MODELS                  ║
╠══════════════════════════════════════════════════════════╣
║                                                        ║
║  Whisper INT8 (merged):    ~2.5 GB                     ║
║  NLLB 4-bit (+ LoRA):     ~0.8 GB                     ║
║  XTTS-v2:                  ~0.8 GB                     ║
║  Emotion Detector:         ~0.4 GB (small Wav2Vec2)    ║
║  CUDA overhead:            ~1.0 GB                     ║
║                                                        ║
║  TOTAL GPU:                ~5.1 GB  ✓                  ║
║  ECAPA-TDNN (CPU):         ~0.3 GB RAM                 ║
║                                                        ║
║  Headroom: 2.5 GB (safety margin)                      ║
╚══════════════════════════════════════════════════════════╝
```

### 8.3 Storage Budget

```
╔══════════════════════════════════════════════════════════╗
║            STORAGE BUDGET — 60 GB Total                ║
╠══════════════════════════════════════════════════════════╣
║                                                        ║
║  OS + Python env + libs:         ~8 GB                 ║
║                                                        ║
║  Datasets (processed):                                 ║
║    CoVoST-2 Telugu:              ~2.5 GB               ║
║    Rasa Telugu:                  ~1.5 GB               ║
║    VCTK:                         ~8.0 GB               ║
║    LJSpeech:                     ~2.5 GB               ║
║    Subtotal:                    ~14.5 GB               ║
║                                                        ║
║  Models (quantized):                                   ║
║    Whisper INT8 + LoRA:          ~1.6 GB               ║
║    ECAPA-TDNN:                   ~0.1 GB               ║
║    IndicTrans2 4-bit:            ~0.8 GB               ║
║    XTTS-v2 (×2 phases):         ~3.6 GB               ║
║    Emotion detector:             ~0.4 GB               ║
║    Subtotal:                     ~6.5 GB               ║
║                                                        ║
║  Working space:                 ~17.1 GB               ║
║  Buffer:                        ~13.9 GB               ║
╚══════════════════════════════════════════════════════════╝
```

---

## 9. Evaluation Framework

### 9.1 Metrics Overview

| Metric | Module | What It Measures | Library | Target |
|---|---|---|---|---|
| **WER** | ASR | Word-level transcription accuracy | `jiwer` | ≤ 20% |
| **CER** | ASR | Character-level transcription accuracy | `jiwer` | ≤ 10% |
| **BLEU** | Translation | N-gram overlap with reference translation | `sacrebleu` | ≥ 15 (pipeline), ≥ 22 (MT-only) |
| **chrF** | Translation | Character-level F-score | `sacrebleu` | — |
| **EER** | Speaker | Equal Error Rate for speaker verification | Custom (ROC) | ≤ 12% |
| **SECS** | TTS + Speaker | Speaker Embedding Cosine Similarity | Custom (ECAPA) | ≥ 0.70 |
| **CCC** | Emotion | Concordance Correlation Coefficient for VAD | Custom | ≥ 0.55 |
| **UTMOS** | TTS | Predicted Mean Opinion Score | UTMOS22 | ≥ 3.5 |
| **Latency** | Pipeline | End-to-end processing time | `time.time()` | ≤ 10s/utterance |

### 9.2 Metric Calculations

**WER (Word Error Rate):**
$$\text{WER} = \frac{S + D + I}{N}$$
where $S$ = substitutions, $D$ = deletions, $I$ = insertions, $N$ = total reference words

**CCC (Concordance Correlation Coefficient):**
$$\text{CCC} = \frac{2 \sigma_{xy}}{\sigma_x^2 + \sigma_y^2 + (\mu_x - \mu_y)^2}$$
where $\sigma_{xy}$ = covariance, $\sigma^2$ = variance, $\mu$ = mean

**BLEU:**
$$\text{BLEU} = \text{BP} \times \exp\left(\sum_{n=1}^{4} \frac{1}{4} \log p_n\right)$$
where $\text{BP}$ = brevity penalty, $p_n$ = $n$-gram precision

**SECS (Speaker Embedding Cosine Similarity):**
$$\text{SECS} = \frac{e_{\text{input}} \cdot e_{\text{output}}}{\|e_{\text{input}}\| \cdot \|e_{\text{output}}\|}$$
where $e$ = 192-dim d-vectors from ECAPA-TDNN

---

## 10. Ablation Studies

Three ablation studies are conducted in NB08 to quantify the contribution of each novel component:

### 10.1 Ablation 1: No FiLM Conditioning

- **Method:** Force VAD vector to [0, 0, 0] (neutral emotion), bypassing emotion detector
- **Measures:** Impact of emotional conditioning on speaker similarity and naturalness
- **Expected:** Lower SECS (less natural speech → less speaker-like)

### 10.2 Ablation 2: No Contrastive Fine-Tuning

- **Method:** Replace fine-tuned ECAPA-TDNN with vanilla pretrained SpeechBrain checkpoint
- **Measures:** Impact of adversarial language-invariant training on cross-lingual voice cloning
- **Expected:** Lower SECS (speaker embedding will encode Telugu language features, causing mismatch with English output)

### 10.3 Ablation 3: Discrete Labels vs Continuous VAD

- **Method:** Round continuous VAD vector to nearest Russell & Mehrabian centroid (8 emotions)
- **Measures:** Information loss from discretizing emotion representation
- **Expected:** Slightly lower quality (quantization of continuous signal)

### Results Format

```
Condition                     SECS
─────────────────────────────────────
Full pipeline                 X.XXXX
No FiLM                       X.XXXX
No contrastive                X.XXXX
Discrete labels               X.XXXX
```

---

## 11. Key Technical Innovations & Bug Fixes

### 11.1 Custom Training Loop for Whisper + PEFT

**Problem:** HuggingFace Trainer + PEFT + BitsAndBytes INT8 causes `decoder_input_ids` errors because multiple wrapper layers (PEFT → Accelerate → DataParallel) interfere with Whisper's unique input format (`input_features` not `input_ids`).

**Solution:** Plain PyTorch training loop with explicit model unwrapping:
```python
# Unwrap all wrappers to get actual PEFT model
actual_model = model
while hasattr(actual_model, 'module'):
    actual_model = actual_model.module

# Call with correct Whisper inputs
outputs = actual_model(
    input_features=mel_features,  # NOT input_ids!
    labels=targets
)
```

### 11.2 LoRA Without `task_type` for Whisper

**Problem:** Setting `task_type=SEQ_2_SEQ_LM` in LoRA config causes PEFT to inject `input_ids` into the encoder, but Whisper expects `input_features` (mel spectrograms).

**Solution:** Omit `task_type` entirely and manually handle the forward pass.

### 11.3 Gradient Checkpointing Disabled for Quantized Models

**Problem:** `prepare_model_for_kbit_training()` enables gradient checkpointing by default, which conflicts with BitsAndBytes quantization causing gradient flow errors.

**Solution:**
```python
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
model.config.use_cache = False
model.gradient_checkpointing_disable()
```

### 11.4 Gradient Reversal Layer for Language-Invariant Embeddings

**Innovation:** The speaker encoder uses a Gradient Reversal Layer (GRL) with an adversarial language classifier:

```
Speaker Embedding
    ├── Speaker Classification Loss (normal gradients → learn speaker identity)
    └── Language Classification Loss (REVERSED gradients → forget language info)
```

The GRL multiplies gradients by $-\lambda$ during backpropagation, forcing the encoder to produce embeddings that **cannot** distinguish between Telugu and English. λ anneals from 0→1 during training (Ganin et al. 2016 schedule).

### 11.5 XTTS-v2 Garbling Fix

**Problem:** Output was "grr grr gha gha" due to aggressive repetition penalty (10.0) and high temperature (0.75).

**Fix:** Switch to low-level API with controlled parameters:
- `repetition_penalty` → 2.0 (from 10.0)
- `temperature` → 0.3 (from 0.75)
- Reference clips → 8-10s LJSpeech (from 3.3s RAVDESS)
- `gpt_cond_len` → 12 (better speaker conditioning)

### 11.6 NLLB Tokenization Fix

**Problem:** `as_target_tokenizer()` context manager is deprecated/broken for NLLB models.

**Fix:** Use `text_target=` keyword argument:
```python
# WRONG: tokenizer.as_target_tokenizer()
# RIGHT:
target_tokens = tokenizer(text_target=english_text, ...)
```

---

## 12. Python Scripts

The project includes 4 standalone Python scripts (currently empty/placeholder):

| Script | Intended Purpose |
|---|---|
| `compute_dvectors.py` | Batch computation of speaker d-vectors for all audio files |
| `finetune_yourtts.py` | YourTTS fine-tuning script (Tier 3 from NB07b) |
| `prosody.py` | Prosody transfer utilities (pitch, rhythm, energy mapping) |
| `tts_engine.py` | Unified TTS inference engine (wraps XTTS-v2/F5-TTS/YourTTS) |

These scripts are scaffolds for modularizing notebook code into production-ready modules.

---

## 13. File & Directory Structure

```
Project/
├── Codes/                                    # Main codebase
│   ├── 01_environment_setup.ipynb            # System checks, deps, configs
│   ├── 02_data_pipeline.ipynb                # Download, preprocess, manifests
│   ├── 03_whisper_asr_finetuning.ipynb       # Whisper LoRA training (Telugu ASR)
│   ├── 04_speaker_encoder_finetuning.ipynb   # ECAPA-TDNN + GRL (CPU)
│   ├── 05_translation_finetuning.ipynb       # NLLB QLoRA (Telugu→English)
│   ├── 06_emotion_detector.ipynb             # Wav2Vec2 → VAD regression
│   ├── 07_tts_finetuning.ipynb               # XTTS-v2 zero-shot cloning
│   ├── 07b_tts_rebuild.ipynb                 # 3-tier TTS fix (XTTS/F5/YourTTS)
│   ├── 08_pipeline_integration_demo.ipynb    # Full pipeline + eval + Gradio
│   ├── compute_dvectors.py                   # D-vector batch computation
│   ├── finetune_yourtts.py                   # YourTTS training script
│   ├── prosody.py                            # Prosody transfer utilities
│   └── tts_engine.py                         # Unified TTS engine
│
├── Review 1/                                 # Documentation (earlier review)
│   ├── ARCHITECTURE_AND_METHODOLOGY.md       # Detailed architecture doc
│   ├── context.md                            # Hybrid workflow guide
│   ├── preprocessing_notebook.ipynb          # Review 1 preprocessing
│   └── training_local_rtx4060.ipynb          # Review 1 training
│
├── configs/                                  # Generated by NB01
│   ├── asr.yaml
│   ├── speaker.yaml
│   ├── translation.yaml
│   ├── emotion.yaml
│   └── tts.yaml
│
├── data/                                     # Generated by NB02
│   ├── raw/                                  # Temporary (deleted after processing)
│   ├── processed/
│   │   ├── asr_train/
│   │   ├── speaker_train/
│   │   ├── tts_train/
│   │   └── emotion_train/
│   ├── metadata/                             # CSV manifests
│   └── embeddings/                           # Pre-computed d-vectors
│
├── checkpoints/                              # Trained model weights
│   ├── whisper_merged/                       # Merged Whisper (INT8 + LoRA)
│   ├── speaker_encoder/                      # ECAPA-TDNN best checkpoint
│   ├── indictrans2_finetuned/                # NLLB LoRA adapter
│   ├── xtts_phase5a/                         # TTS Phase 5A checkpoint
│   ├── xtts_phase5b/                         # TTS Phase 5B checkpoint
│   └── emotion_detector/                     # Wav2Vec2 + VAD head
│
├── results/                                  # Evaluation outputs
│   ├── evaluation_results.json
│   ├── ablation_results.json
│   └── sample_outputs/                       # Generated audio samples
│
├── demo/                                     # Gradio demo files
├── logs/                                     # Training logs (CSV)
│
└── PROJECT_DOCUMENTATION.md                  # ← THIS FILE
```

---

## 14. Target Metrics vs Achieved Results

### Target Metrics

| Component | Metric | Target |
|---|---|---|
| **ASR (Whisper)** | WER | ≤ 20% |
| **ASR (Whisper)** | CER | ≤ 10% |
| **Speaker Encoder** | EER | ≤ 12% |
| **Speaker Encoder** | Cosine Sim (same-speaker) | ≥ 0.70 |
| **Translation (NLLB)** | BLEU (MT-only) | ≥ 22 |
| **Translation (NLLB)** | BLEU (pipeline) | ≥ 15 |
| **Emotion Detector** | CCC | ≥ 0.55 |
| **TTS (XTTS-v2)** | SECS | ≥ 0.70 |
| **TTS (XTTS-v2)** | UTMOS | ≥ 3.5 |
| **Pipeline** | Latency | ≤ 10s / utterance |
| **Pipeline** | Total GPU VRAM | ≤ 8 GB |

### Actual Results

Results depend on the specific training run and are stored in:
- `results/evaluation_results.json` — Component and pipeline metrics
- `results/ablation_results.json` — Ablation study comparisons
- `logs/` — Per-step training logs (loss, WER, CER, BLEU, LR, VRAM)

---

## 15. Future Enhancements

### 15.1 Model Improvements

| Enhancement | Description |
|---|---|
| **Streaming ASR** | Chunk audio in real-time for progressive transcription |
| **Model Distillation** | Train smaller student to mimic Whisper-large (4x smaller, 2-3x faster) |
| **Joint Training** | Audio → English directly (bypass intermediate Telugu text) |
| **Larger LoRA Rank** | Increase from r=8 to r=32/64 for better adaptation capacity |

### 15.2 Data Enhancements

| Enhancement | Description |
|---|---|
| **SpecAugment** | Already implemented — mask time/frequency bins |
| **Speed Perturbation** | Already implemented — 0.9×/1.0×/1.1× |
| **Noise Injection** | Add real-world background noise (cafes, traffic) |
| **Domain Adaptation** | Fine-tune for specific domains (medical, legal, technical) |

### 15.3 System Enhancements

| Enhancement | Description |
|---|---|
| **Confidence Scoring** | Flag low-confidence predictions for human review |
| **Multi-language Support** | Extend to other Indian languages (Whisper + NLLB support 200+ languages) |
| **Web Deployment** | FastAPI server + React/Streamlit frontend |
| **Batch Processing** | Queue system for processing multiple files |

---

## 16. References

### Models

| Model | Paper | Authors |
|---|---|---|
| **Whisper** | "Robust Speech Recognition via Large-Scale Weak Supervision" | Radford et al. (OpenAI, 2022) |
| **NLLB-200** | "No Language Left Behind: Scaling Human-Centered Machine Translation" | Costa-jussà et al. (Meta AI, 2022) |
| **ECAPA-TDNN** | "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification" | Desplanques et al. (2020) |
| **Wav2Vec2** | "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations" | Baevski et al. (Facebook AI, 2020) |
| **XTTS-v2** | Coqui TTS — Open-source TTS engine | Coqui AI |

### Techniques

| Technique | Paper | Authors |
|---|---|---|
| **LoRA** | "LoRA: Low-Rank Adaptation of Large Language Models" | Hu et al. (Microsoft, 2021) |
| **QLoRA** | "QLoRA: Efficient Finetuning of Quantized LLMs" | Dettmers et al. (UW, 2023) |
| **FiLM** | "FiLM: Visual Reasoning with a General Conditioning Layer" | Perez et al. (2018) |
| **GRL** | "Domain-Adversarial Training of Neural Networks" | Ganin et al. (2016) |
| **AAM-Softmax** | "ArcFace: Additive Angular Margin Loss for Deep Face Recognition" | Deng et al. (2019) |

### Frameworks

| Framework | URL |
|---|---|
| HuggingFace Transformers | https://huggingface.co/docs/transformers |
| PEFT | https://huggingface.co/docs/peft |
| BitsAndBytes | https://github.com/TimDettmers/bitsandbytes |
| SpeechBrain | https://speechbrain.github.io |
| Coqui TTS | https://github.com/coqui-ai/TTS |

---

**Document Version:** 2.0  
**Project:** TeluguVoiceBridge v2  
**Hardware:** RTX 4060 Laptop (7.6 GB VRAM, 24 GB RAM)  
**Total Notebooks:** 9 (01–08, including 07b)  
**Total Models:** 5 (Whisper + ECAPA-TDNN + NLLB + Wav2Vec2 + XTTS-v2)  
**Total Training Time:** ~12-17 hours (all phases, sequential)  
**Inference VRAM:** ~5.1 GB (all 5 models loaded simultaneously)
