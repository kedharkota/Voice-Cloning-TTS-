# TeluguVoiceBridge — Architecture, Workflow & Methodology

> Complete technical breakdown of a Telugu → English speech-to-speech translation system that **preserves the speaker's voice, emotion, and prosody** across languages.

---

## Table of Contents

1. [What Is This Project?](#1-what-is-this-project)
2. [The Core Problem — Why Voice Preservation Is Hard](#2-the-core-problem)
3. [High-Level Pipeline Overview](#3-high-level-pipeline-overview)
4. [Component 1 — ASR: Whisper Large-v3](#4-component-1--asr-whisper-large-v3)
5. [Component 2 — Speaker Encoder: ECAPA-TDNN](#5-component-2--speaker-encoder-ecapa-tdnn)
6. [Component 3 — Emotion Detector: Wav2Vec2 VAD](#6-component-3--emotion-detector-wav2vec2-vad)
7. [Component 4 — Translation: NLLB-600M + LoRA](#7-component-4--translation-nllb-600m--lora)
8. [Component 5 — TTS: YourTTS (VITS)](#8-component-5--tts-yourtts-vits)
9. [Component 6 — Voice Conversion: FreeVC24](#9-component-6--voice-conversion-freevc24)
10. [Component 7 — Prosody Transfer](#10-component-7--prosody-transfer)
11. [Component 8 — FiLM Emotion Conditioning](#11-component-8--film-emotion-conditioning)
12. [D-Vectors: How Voice Identity Is Stored](#12-d-vectors-how-voice-identity-is-stored)
13. [Training the TTS — YourTTS Fine-Tuning](#13-training-the-tts--yourtts-fine-tuning)
14. [SECS: How Voice Similarity Is Measured](#14-secs-how-voice-similarity-is-measured)
15. [Full Data Flow — Step by Step](#15-full-data-flow--step-by-step)
16. [Memory & Hardware Layout](#16-memory--hardware-layout)
17. [File Structure Reference](#17-file-structure-reference)
18. [Datasets Used](#18-datasets-used)
19. [Key Design Decisions & Why](#19-key-design-decisions--why)
20. [Evaluation Metrics](#20-evaluation-metrics)
21. [Glossary of Terms](#21-glossary-of-terms)

---

## 1. What Is This Project?

This is a **speech-to-speech translation** system. You speak a sentence in **Telugu** into a microphone. The system outputs the **same sentence spoken in English**, but in **your voice** — preserving your unique vocal timbre, your emotional tone, and your natural speech rhythm.

### What "voice preservation" means concretely

When you say something sad in Telugu, the English output should also sound sad. When you speak with a deep baritone voice, the English output should also have a deep baritone. When you speak quickly and with emphasis on certain words, the English output should match that rhythm.

Most translation systems just output a generic robotic TTS voice. This project tries to make the output sound like *you* said it in English.

---

## 2. The Core Problem

Voice preservation across languages is not solved by simply "feeding the speaker embedding to TTS." There are five fundamental challenges:

### Problem 1: Speaker Encoder Language Leakage
Most speaker encoders (including the popular Resemblyzer) were trained on monolingual English data. When you feed them Telugu audio, the embedding they produce mixes up **who the speaker is** with **what language they're speaking**. The Telugu phoneme distribution looks like a different speaker entirely to these models.

**Fix:** We use ECAPA-TDNN fine-tuned for language-agnostic speaker separation, plus post-hoc voice conversion (FreeVC24) as a second-stage correction.

### Problem 2: Prosody Collapse at the Text Boundary
When you translate text from Telugu to English, all timing information is lost. A Telugu sentence spoken fast with rising intonation on the word "where?" becomes a flat English string. The TTS has no idea that "where?" was emphasized.

**Fix:** Prosody extraction runs on the input Telugu audio (F0 contour, energy envelope, speech rate), and cross-attention from the translation model is used to *align* Telugu prosody features with their corresponding English words.

### Problem 3: Emotion Erasure
Standard TTS produces emotionally neutral speech. The input might be angry, joyful, or fearful, but the output is always calm.

**Fix:** An emotion detector extracts a 3-dimensional VAD vector (Valence, Arousal, Dominance). This vector is injected into the TTS decoder via FiLM conditioning layers that modulate the waveform generation process.

### Problem 4: TTS Generalization (Zero-Shot Voice Cloning)
YourTTS pretrained model was only trained on 6 speakers. When you give it a speaker embedding from a Telugu speaker it has never seen, it produces an "average voice" that ignores the conditioning. Zero-shot voice cloning only works well when the TTS has been trained on hundreds of diverse speakers.

**Fix:** We fine-tune YourTTS on ~2,212 speakers from LibriTTS + the Speech Accent Archive (2,133 speakers, 200 native languages).

### Problem 5: Cross-Lingual Embedding Space Mismatch in Metrics
When measuring voice similarity (SECS — Speaker Encoder Cosine Similarity), if you use a *different* encoder to measure similarity than the one the TTS uses for conditioning, you get misleading scores. ECAPA-TDNN (192-dim) and YourTTS's internal encoder (512-dim) live in completely different embedding spaces.

**Fix:** We use YourTTS's own speaker encoder for the SECS metric, so the measurement and the conditioning use the same embedding space.

---

## 3. High-Level Pipeline Overview

```
INPUT: Telugu speech audio  (wav, 16kHz mono)
         │
         │
         ├──────────────────────────────────────────────────────────────────┐
         │                                                                  │
         ▼                                                                  │
  ┌─────────────┐                                                           │
  │  1. WHISPER │  Telugu speech → Telugu text                              │
  │  ASR (INT8) │  (Automatic Speech Recognition)                           │
  └──────┬──────┘                                                           │
         │ "నేను ఇంట్లో ఉన్నాను"                                           │
         ▼                                                                  │
  ┌─────────────────┐                                                       │
  │  2. NLLB-600M   │  Telugu text → English text                           │
  │  + LoRA adapter │  + cross-attention matrix for prosody alignment       │
  └──────┬──────────┘                                                       │
         │ "I am at home."                                                  │
         │                                                                  │
         │          ┌─────────────────────────────────────────┐            │
         │          │  3. ECAPA-TDNN Speaker Encoder (CPU)    │◄───────────┤
         │          │  192-dim speaker identity vector        │            │
         │          └─────────────────────────────────────────┘            │
         │                                                                  │
         │          ┌─────────────────────────────────────────┐            │
         │          │  4. Wav2Vec2 Emotion Detector (GPU)     │◄───────────┤
         │          │  3-dim VAD: [Valence, Arousal, Dominance]│           │
         │          └─────────────────────────────────────────┘            │
         │                                                                  │
         │          ┌─────────────────────────────────────────┐            │
         │          │  5. Prosody Extractor (CPU, praat)      │◄───────────┘
         │          │  F0 contour, energy, speech rate        │
         │          └─────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
  ┌──────────────────────────────────────────────────────────┐
  │        6. YourTTS (VITS) — Text-to-Speech Engine         │
  │   Conditions on:                                         │
  │     • English text tokens                                │
  │     • 512-dim d-vector (speaker identity)                │
  │     • FiLM modulation from VAD vector (emotion)          │
  │     • Prosody targets from cross-attention alignment     │
  │   Outputs: English speech waveform @ 16kHz               │
  └──────────────────────────────┬───────────────────────────┘
                                 │
                                 ▼
  ┌──────────────────────────────────────────────────────────┐
  │        7. FreeVC24 Voice Conversion                       │
  │   Takes: YourTTS output + original Telugu audio           │
  │   Applies waveform-level speaker matching                │
  │   Output: 24kHz English speech with better voice match   │
  └──────────────────────────────┬───────────────────────────┘
                                 │
                                 ▼
  OUTPUT: English speech with cloned voice @ 24kHz
```

---

## 4. Component 1 — ASR: Whisper Large-v3

### What it does
Converts Telugu speech audio into Telugu text.

### Model
**OpenAI Whisper Large-v3** — a transformer encoder-decoder model with ~1.5B parameters. It was pre-trained on 680,000 hours of multilingual audio from the internet. It natively supports Telugu (language code: `te`).

### How a transformer encoder-decoder works (basic)
1. **Encoder:** The audio waveform is converted to a Mel spectrogram (a 2D time-frequency representation). This is passed through several transformer layers that learn to detect phonemes, words, and audio patterns. Output: a sequence of hidden state vectors representing the audio.
2. **Decoder:** Given the encoder output, a language model generates text tokens one by one (auto-regressive). Each new token attends to all encoder states (via cross-attention) to decide what word comes next.

### Fine-tuning
We fine-tuned Whisper on the **FLEURS Telugu dataset** + **CommonVoice Telugu** to improve accuracy on Telugu-specific accents and vocabulary. The fine-tuned weights are stored in `checkpoints/whisper_merged/merged/`.

### Quantization (INT8)
The model is loaded in **8-bit integer quantization** using `bitsandbytes`. This means each model weight (normally a 32-bit float taking 4 bytes) is stored as an 8-bit integer (1 byte). The model shrinks from ~6 GB to ~1.5 GB in GPU memory. Tiny quality loss, 4× memory savings.

### Long audio handling
Whisper's feature extractor only processes **30 seconds** at a time. For longer audio:
- Split into 28-second chunks with 3-second overlap
- Transcribe each chunk
- Deduplicate the overlap by matching word sequences at chunk boundaries
- Concatenate the unique portions

### Code location
- Loading: `app.py` lines ~85–105 (`load_all_models`)
- Inference: `app.py` `transcribe_telugu()` function

---

## 5. Component 2 — Speaker Encoder: ECAPA-TDNN

### What it does
Takes Telugu audio and produces a compact vector (192 numbers) that captures **who** the speaker is — their unique vocal "fingerprint." This is called a **speaker embedding** or **d-vector**.

### Model
**ECAPA-TDNN** (Emphasized Channel Attention, Propagation and Aggregation — Time Delay Neural Network) from SpeechBrain, trained on VoxCeleb for speaker verification.

### How ECAPA-TDNN works (basic)
A TDNN looks at not just the current audio frame but also frames from the past and future simultaneously (like looking at a window of audio). ECAPA adds:
- **Channel attention:** Weights different frequency channels by importance
- **Multi-scale feature aggregation:** Combines features from different temporal scales
- **Attentive statistics pooling:** Creates a fixed-size vector from variable-length audio by computing a weighted mean and standard deviation

The output is a 192-dimensional "speaker embedding" — a point in 192D space where all utterances from the same speaker cluster together, regardless of what words are spoken.

### Fine-tuning
Fine-tuned with **additive angular margin loss (ArcFace)** on Telugu speaker data to reduce language leakage. The fine-tuned weights are in `checkpoints/speaker_encoder/best_model.ckpt`.

### Attentive segment pooling (our improvement)
Instead of encoding the full audio in one pass (which is noisy), we:
1. Split audio into **1.5-second segments with 0.5-second hop** (67% overlap)
2. L2-normalize each segment's embedding
3. Compute **energy-weighted average** — louder segments (more speech content) contribute more than quiet segments
4. Final L2-normalize the result

This produces a more stable speaker embedding that's less affected by noise, pauses, and disfluencies.

### Code location
- Loading: `app.py` lines ~110–130
- Inference: `app.py` `extract_speaker_embedding()` function

---

## 6. Component 3 — Emotion Detector: Wav2Vec2 VAD

### What it does
Takes audio and predicts a **3-dimensional emotion vector**: [Valence, Arousal, Dominance].

- **Valence:** Positive/negative emotion. +1 = very happy. -1 = very sad.
- **Arousal:** Energy level. +1 = very excited/angry. -1 = calm/sleepy.
- **Dominance:** Control/confidence. +1 = dominant/commanding. -1 = submissive/timid.

This is called the **VAD model of emotion** and is more useful than discrete labels (angry/happy/sad) because it's continuous and can represent nuanced emotional states.

### Model Architecture
```
Wav2Vec2-base backbone (Facebook, 94M params)
  ↓  CNN feature extractor (frozen)
  ↓  Transformer encoder (first 6 layers frozen, top 6 trainable)
  ↓  Mean pooling over time
  ↓  Dropout(0.1) → Linear(768→128) → ReLU → Dropout(0.1) → Linear(128→3) → Tanh
  ↓  Output: [valence, arousal, dominance] ∈ [-1, 1]³
```

### How Wav2Vec2 works (basic)
Wav2Vec2 is a self-supervised model. It was pre-trained on raw audio by:
1. Masking random parts of the audio
2. Training to predict the masked parts from context

This forces it to learn rich acoustic representations of speech without needing labels. The CNN feature extractor converts raw waveform → frame features at 20ms intervals. The transformer then looks at the full sequence context.

### Training data
Trained on the **EmoV-DB** emotional speech database on the regression task of predicting continuous VAD scores.

### Code location
- Loading: `app.py` lines ~150–185
- Inference: `app.py` `extract_emotion_vad()` function

---

## 7. Component 4 — Translation: NLLB-600M + LoRA

### What it does
Translates Telugu text into English text, AND extracts the **cross-attention matrix** used for prosody alignment.

### Model
**NLLB-200 (No Language Left Behind)** — Facebook's multilingual translation model supporting 200 languages. We use the 600M parameter distilled version.

### How seq2seq translation works (basic)
1. Tokenize the Telugu text into sub-word tokens using a byte-pair encoding (BPE) vocabulary
2. Set the source language token: `tel_Telu`
3. Encoder processes all source tokens at once → hidden states
4. Decoder generates target tokens one by one (forced first token: `eng_Latn`)
5. At each decoder step, **cross-attention** shows which source tokens the decoder is looking at

### LoRA Fine-tuning
LoRA (Low-Rank Adaptation) is a technique to fine-tune large models efficiently:
- Instead of updating all 600M weights, we **add small matrices** (rank ~16) to specific attention layers
- These small additions have ~1M parameters total vs 600M original
- Training is much faster and uses far less memory
- The LoRA adapter is saved in `checkpoints/indictrans2_finetuned/best_lora/`

Fine-tuned on **CoVoST-2 Telugu→English** translation pairs to improve Telugu-specific vocabulary and idioms.

### 4-bit quantization (NF4)
The base NLLB model is loaded in **4-bit NF4 quantization** (Normal Float 4). Each weight stored in 4 bits instead of 32 bits. 8× memory reduction. The compute happens in float16.

### Cross-attention extraction
During inference, we save the cross-attention matrix from the **last decoder layer**:
- Shape: `[n_english_tokens, n_telugu_tokens]`
- `attention[i][j]` = how much the i-th English token attends to the j-th Telugu token
- This is used to **align prosodic features** between languages (see Component 7)

### Long text handling
For text over 200 characters, split on sentence boundaries (`।`, `.`, `!`, `?`) and translate each sentence separately to avoid tokenizer truncation.

### Code location
- Loading: `app.py` lines ~135–155
- Inference: `app.py` `translate_te_en()` function

---

## 8. Component 5 — TTS: YourTTS (VITS)

### What it does
Takes English text + a speaker d-vector and generates English speech audio that sounds like the target speaker.

### Model: VITS (Variational Inference with adversarial learning for end-to-end TTS)

VITS is an end-to-end TTS model — it goes directly from text to waveform without an intermediate spectrogram step. Architecture:

```
Text tokens
    │
    ▼
Text Encoder (Transformer)
    │  Phoneme/character representations
    ▼
Stochastic Duration Predictor ──┐
    │                            │ duration for each phoneme
Prior Distribution (Flow)       │
    │  Latent z                  │
    ▼                            │
Posterior Flow (during training)│
    │                            │
Variational Autoencoder         │
    │  Latent z conditioned on   │
    │  speaker d-vector          │
    ▼                            ▼
HiFi-GAN Decoder ← alignment from duration predictor
    │  ResBlock stack (progressive upsampling)
    │  Each ResBlock: dilated convolutions
    ▼
Waveform output @ 16kHz
```

**The key insight:** Speaker conditioning (your voice) is injected via a **d-vector** — a 512-dimensional vector that captures your vocal characteristics. This vector is concatenated to the latent representation at multiple points in the network.

### YourTTS specifics
YourTTS extends VITS with:
- **Multi-speaker** support via d-vector conditioning
- **Multi-lingual** support (English, French, Portuguese pre-trained)
- **Zero-shot voice cloning:** give it any speaker's audio at inference time

### Fine-tuning
The pretrained YourTTS only knows 6 speakers. We fine-tune on:
- **LibriTTS dev-clean + test-clean:** 79 speakers, ~10,573 clips, diverse English text
- **Speech Accent Archive:** 2,133 speakers, 1 clip each, all reading the same passage, 200 native languages

Total: **2,212 speakers**. Training on this many diverse speakers dramatically improves the model's ability to generalize to **unseen speakers** at inference time.

### Reference audio for voice cloning
At inference, YourTTS needs a short reference audio clip from the target speaker (3–10 seconds). It passes this through its own internal speaker encoder to get a 512-dim d-vector, then conditions the entire synthesis on this vector.

### Code location
- Loading: `app.py` lines ~187–220
- Inference: `app.py` `telugu_voice_bridge()` function

---

## 9. Component 6 — Voice Conversion: FreeVC24

### What it does
A two-stage approach to voice cloning:
1. **Stage 1 (YourTTS):** Generate English speech conditioned on speaker d-vector
2. **Stage 2 (FreeVC24):** Transform the output waveform to match the target voice at a deeper acoustic level

### Why two stages?
YourTTS operates at the **latent code level** — the voice conditioning influences the statistical distribution of the generated latent representations. But residual voice information from the "average training voice" bleeds through.

FreeVC24 operates at the **waveform level** — it applies voice conversion directly on the audio signal. It can fix things YourTTS got wrong: formant positions, vocal tract length, breathiness, accent-specific phoneme articulation.

### How FreeVC24 works (basic)
FreeVC24 disentangles speech into:
- **Content:** WHAT is being said (phoneme sequence) — from a WavLM encoder
- **Speaker:** WHO is saying it (from a speaker encoder)

Then it re-synthesizes the speech with the same content but the target speaker's voice characteristics. The key is that content and speaker are encoded independently, so swapping the speaker doesn't change the words.

### Output sample rate
FreeVC24 outputs at **24kHz** (vs Whisper/YourTTS's 16kHz). Higher quality for the final output.

### Code location
- Loading: `app.py` lines ~222–240
- Applied in: `app.py` `telugu_voice_bridge()` function

---

## 10. Component 7 — Prosody Transfer

### What it does
Extracts prosodic features from the Telugu input and transfers them to the English output, so the rhythm and emphasis pattern of the original speech is preserved.

### What is prosody?
Prosody is the "music" of speech — the variation in:
- **Pitch (F0):** How high or low your voice is (falling pitch = statement, rising = question)
- **Energy:** How loud or soft different parts are (louder = more emphasis)
- **Duration:** How long each word/syllable takes (slow speech vs fast)
- **Rhythm:** The pattern of stressed and unstressed syllables

Translation converts words but **destroys prosody** — a fast excited question in Telugu becomes a flat declarative sentence in English.

### F0 extraction (via Praat)
We use **Praat** (via `praat-parselmouth`) to extract:
1. F0 (fundamental frequency) contour at 10ms intervals
2. Convert to **semitones** relative to the speaker's own mean F0 (normalizes for different voice registers)
3. Energy (RMS) at 10ms intervals, normalized to speaker mean

### Cross-lingual alignment via MT attention
The translation model's cross-attention matrix `[n_eng_tokens × n_tel_tokens]` tells us which English words correspond to which Telugu words. Using this:
1. Compute **word-level prominence** in Telugu (peak F0 + energy for each word)
2. Map to corresponding English words via the attention matrix
3. Pass prominence targets to the TTS as soft constraints on duration and pitch

### Global Style Tokens (GST)
A reference encoder processes the input Mel spectrogram and produces a **style embedding** that captures utterance-level speaking style (fast/slow, expressive/monotone, formal/casual). This is blended into the TTS conditioning.

### Code location
`prosody.py` — `ProsodyExtractor`, `ProsodyEncoder`, `summarize_prosody_for_tts()`

---

## 11. Component 8 — FiLM Emotion Conditioning

### What it does
Injects the emotion VAD vector into the TTS decoder's waveform generation process via **FiLM (Feature-wise Linear Modulation)** layers.

### Why hook into the decoder?
Normal TTS conditioning (feeding emotion as a vector) only affects the high-level latent representation. But fine-grained prosodic details (micro-variations in pitch, breathiness, voice tremor) happen in the final waveform generation layers. FiLM hooks directly into the HiFi-GAN decoder's ResBlocks.

### How FiLM works
FiLM applies a **learned linear transformation** to each feature map in the decoder:

```
output = γ(emotion) * feature_map + β(emotion)

where:
  γ (gamma): scale factor — learned from VAD vector
  β (beta):  shift factor — learned from VAD vector
  feature_map: activation at that layer of the decoder
```

γ and β are produced by a small MLP: `VAD(3-dim) → Linear(3→64) → ReLU → Linear(64→2×channels) → split γ, β`

Initialized to γ=1, β=0 (identity — no effect until trained).

### Where it's hooked
Registered as **forward hooks** on every 2nd ResBlock of the HiFi-GAN decoder. This means the emotion modulation runs automatically during every TTS inference call, without modifying TTS source code.

### Code location
- FiLM layer: `src/tts/film.py`
- Hook setup: `app.py` lines ~250–310
- VAD vector injection: `app.py` in `telugu_voice_bridge()` — sets `M["_vad_vector"]` before calling TTS

---

## 12. D-Vectors: How Voice Identity Is Stored

### What is a d-vector?
A d-vector (discriminative vector) is the compact numerical representation of a speaker's voice identity. In YourTTS, it's a 512-dimensional float vector.

Think of it like a GPS coordinate in 512-dimensional "voice space." Every person has a unique location in this space. Speakers with similar voice characteristics (similar accent, similar pitch range) are close to each other. Very different voices are far apart.

### How d-vectors are computed
1. Take 1–10 seconds of the speaker's audio
2. Pass through YourTTS's internal speaker encoder (`model_se.pth`)
3. The encoder is a ResNet-style model trained with GE2E loss to make all utterances from the same speaker cluster together
4. Output: a 512-dimensional L2-normalized vector

### Pre-computed d-vector database
We pre-compute d-vectors for all 12,706 training audio clips and store them in `data/dvectors/speakers.json`:
```json
{
  "libritts_devclean#84/121123/84_121123_000007_000001": [0.012, -0.034, ...],
  "accent_archive#afrikaans1": [0.091, 0.045, ...],
  ...
}
```

During training, the model looks up the d-vector for each audio clip by its key and uses it as the conditioning signal. This saves recomputing it every time.

### Key format
- LibriTTS: `libritts_devclean#speaker_id/chapter_id/filename`
- Accent Archive: `accent_archive#language+speaker_id`

### Code location
`compute_dvectors.py` — processes all training data and writes `speakers.json`

---

## 13. Training the TTS — YourTTS Fine-Tuning

### Why fine-tune?
The pretrained YourTTS model was trained on only 6 speakers. Zero-shot voice cloning (cloning ANY unseen voice) requires the model to have **seen thousands of diverse speakers** during training so it generalizes well to new ones.

### Training objective (VITS)
VITS uses a combination of losses:

| Loss | What it penalizes | Weight |
|------|------------------|--------|
| **Mel reconstruction** | Difference between generated and real mel spectrogram | High |
| **KL divergence** | Prior vs posterior latent distribution | Medium |
| **Duration** | Predicted vs actual phoneme durations | Low |
| **Feature matching** | Discriminator intermediate feature difference | Medium |
| **Adversarial (GAN)** | Generator tricking the discriminator | High |
| **Speaker encoder** | Generated speech speaker embedding vs reference | Medium |

The speaker encoder loss is critical for voice cloning — it forces the generated speech to actually sound like the conditioned speaker.

### Training data
```
LibriTTS dev-clean:   40 speakers, 5,736 clips (diverse English text)
LibriTTS test-clean:  39 speakers, 4,837 clips (diverse English text)
Accent Archive:     2,133 speakers, 2,133 clips (one passage, 200 accents)
─────────────────────────────────────────────────────
Total:              2,212 speakers, 12,706 clips
After length filter: ~5,210 training samples per epoch
```

### Hardware constraints & settings
```
GPU:              RTX 4060 8GB VRAM
batch_size:       1  (OOM at batch_size=2 with speaker encoder loss)
max_audio:        7 seconds (clips >7s filtered out to fit VRAM)
learning_rate:    5e-5 (lower than original for stable fine-tuning)
mixed_precision:  True (fp16 for most operations, fp32 for sensitive parts)
grad_accum:       1 (VITS dual-optimizer doesn't support >1)
optimizer:        AdamW (betas=[0.8, 0.99], weight decay=0.01)
lr_scheduler:     ExponentialLR (gamma=0.999875 ≈ very gradual decay)
PYTORCH_CUDA_ALLOC_CONF: expandable_segments=True (avoids fragmentation OOM)
```

### What "restore from pretrained" means
We start from the 6-speaker pretrained YourTTS weights:
- The text encoder, flow, prior network, duration predictor → keep almost entirely (good phoneme knowledge)
- Language embedding layer (`emb_l.weight`) → re-initialized (new language dimensions)
- Discriminator conv_post layers → re-initialized (weight norm format changed in PyTorch 2.6)
- Everything else → fine-tuned from pretrained start

### Training script
`finetune_yourtts.py` — sets up datasets, config, d-vector staging, registers custom formatter, starts Coqui TTS Trainer.

---

## 14. SECS: How Voice Similarity Is Measured

### What is SECS?
**Speaker Encoder Cosine Similarity** — measures how similar two speech samples sound in terms of speaker identity.

```
SECS = cosine_similarity(embed(generated_speech), embed(input_speech))
     = (A · B) / (|A| × |B|)
     ∈ [-1, 1]   (1 = identical voice, 0 = unrelated, -1 = maximally different)
```

### Why SECS was wrong before
We were computing SECS using ECAPA-TDNN's 192-dim embeddings. But YourTTS was conditioned using its own 512-dim embeddings. These live in **different embedding spaces** — cosine similarity between them is meaningless, like measuring distance in meters vs degrees.

The measurement showed 0.38 not because the voice cloning was bad, but because we were comparing apples to oranges.

### The fix
Both the conditioning d-vector AND the SECS metric now use **YourTTS's own speaker encoder** (`model_se.pth` / `config_se.json`). The same 512-dim space is used for everything. The score now actually measures what YourTTS is trying to optimize.

### Target score
SECS > 0.7 is considered good voice cloning. SECS > 0.85 is excellent.

---

## 15. Full Data Flow — Step by Step

Here's exactly what happens when you upload a 10-second Telugu audio clip:

```
1. Audio arrives as numpy array @ 16kHz mono

2. ASR (Whisper):
   - Input:  audio_np [160000 samples]
   - Mel spectrogram extraction → [80, 3000] feature grid
   - INT8 Whisper encoder + decoder
   - Output: "నేను రేపు పొద్దున పని కి వెళ్తాను" (Telugu text)

3. Speaker embedding (ECAPA-TDNN, CPU):
   - Split audio into 1.5s segments with 0.5s hop
   - Encode each segment → 192-dim vector
   - Energy-weighted average → final 192-dim speaker embedding
   - (Used for SECS measurement + as fallback)

4. Emotion detection (Wav2Vec2, GPU):
   - Input: audio_np tensor
   - Wav2Vec2 backbone → 768-dim pooled representation
   - Head: → [valence=0.3, arousal=-0.2, dominance=0.5] (example)
   - Stored in M["_vad_vector"] for FiLM conditioning

5. Prosody extraction (Praat, CPU):
   - F0 contour (pitch over time)
   - Energy envelope (loudness over time)
   - Speaking rate estimate
   - Summarized to: {mean_f0_semitones, f0_range, energy_mean, speaking_rate}

6. Translation (NLLB, GPU):
   - Tokenize Telugu → "tel_Telu" token + 23 tokens
   - 4-bit NLLB encoder-decoder with LoRA
   - 4-beam beam search
   - Output: "I will go to work tomorrow morning." (English text)
   - Cross-attention matrix: [n_eng_tokens × n_tel_tokens]

7. Reference audio extraction:
   - Find highest-energy 7-second segment from input audio
   - This is the "speaker_wav" passed to YourTTS

8. YourTTS synthesis (GPU):
   - YourTTS encodes reference audio → 512-dim d-vector
   - M["_vad_vector"] = [0.3, -0.2, 0.5] already set for FiLM hooks
   - Text encoding → phoneme representations
   - Flow + VAE → latent z conditioned on d-vector
   - HiFi-GAN decoder → waveform
   - FiLM hooks fire at each 2nd ResBlock → emotion modulation applied
   - Output: English waveform @ 16kHz

9. RMS normalization:
   - Scale output to target_rms = 0.08 (LJSpeech level)
   - Peak limit to 0.95 to prevent clipping

10. FreeVC24 voice conversion (GPU):
    - Input: YourTTS output + original Telugu audio
    - WavLM extracts content features from YourTTS output
    - Speaker encoder extracts identity from Telugu audio
    - Decoder re-synthesizes with transferred content + target voice
    - Output: English waveform @ 24kHz

11. SECS computation:
    - YourTTS speaker encoder (model_se.pth) encodes Telugu input → 512-dim
    - Same encoder encodes FreeVC24 output → 512-dim
    - cosine_similarity → SECS score

12. Return to user:
    - Audio: 24kHz English wav
    - Telugu transcript
    - English text
    - SECS score
    - Emotion VAD values
    - Latency breakdown
```

---

## 16. Memory & Hardware Layout

### GPU (RTX 4060, 8GB VRAM)
```
Whisper Large-v3 INT8:  ~1.5 GB
NLLB-600M 4-bit NF4:    ~0.5 GB
Wav2Vec2 emotion:       ~0.4 GB
YourTTS VITS:           ~0.3 GB
FreeVC24:               ~0.3 GB
FiLM layers:            ~0.01 GB
─────────────────────────────
Peak during inference:  ~4.0 GB (models overlap on GPU time)
During TTS training:    ~1.7 GB (only TTS on GPU)
```

### CPU (for inference)
```
ECAPA-TDNN speaker encoder:   CPU only (avoids GPU-CPU transfer overhead)
Prosody extractor (Praat):    CPU only (pure signal processing)
YourTTS SECS encoder:         CPU only (only for metric computation)
```

### Why some models on CPU?
The speaker encoder and prosody extractor run **in parallel** with GPU inference. By keeping them on CPU, they don't compete for GPU memory and can run during the GPU-idle periods.

---

## 17. File Structure Reference

```
pipeline_v2/
│
├── app.py                    # Main web app — all inference code, Gradio UI
├── prosody.py                # Prosody extraction and transfer module
├── tts_engine.py             # Standalone TTS wrapper (used by notebooks)
├── compute_dvectors.py       # Pre-compute d-vectors for all training audio
├── finetune_yourtts.py       # YourTTS fine-tuning script
├── prepare_archive.py        # Convert Speech Accent Archive MP3 → WAV
│
├── configs/
│   ├── asr_config.yaml       # Whisper training config
│   ├── emotion_config.yaml   # VAD model config
│   ├── speaker_config.yaml   # Speaker encoder training config
│   ├── tts_config.yaml       # TTS training overrides
│   └── pipeline_config.yaml  # Full pipeline settings
│
├── data/
│   ├── dvectors/
│   │   └── speakers.json         # Pre-computed d-vectors (138 MB, 12,706 embeddings)
│   ├── raw/
│   │   ├── libritts_clean100/    # LibriTTS dev-clean + test-clean
│   │   ├── archive/              # Speech Accent Archive
│   │   │   ├── wav_16k/          # 2,133 converted WAV files
│   │   │   ├── metadata.csv      # wav_stem|transcript|speaker_name
│   │   │   └── recordings/       # Original MP3 files
│   │   ├── common_voice_te/      # Telugu ASR training data
│   │   ├── fleurs_te/            # FLEURS Telugu
│   │   └── emovdb/               # EmoV-DB for emotion training
│   └── processed/
│       └── tts_train/            # LJSpeech reference audio clips
│
├── checkpoints/
│   ├── whisper_merged/merged/    # Fine-tuned Whisper weights
│   ├── speaker_encoder/          # Fine-tuned ECAPA-TDNN
│   │   ├── best_model.ckpt
│   │   └── pretrained/           # SpeechBrain cache
│   ├── emotion_detector/
│   │   └── best_model.pt         # Fine-tuned Wav2Vec2 VAD
│   ├── indictrans2_finetuned/
│   │   └── best_lora/            # NLLB LoRA adapter
│   ├── yourtts_finetuned/
│   │   └── vits_tts-*/           # Active training run
│   │       ├── best_model.pth    # Best checkpoint (saved when loss improves)
│   │       ├── checkpoint_*.pth  # Regular checkpoints (every 1000 steps)
│   │       └── config.json       # Training config snapshot
│   └── yourtts_restore/          # Staging directory for restore
│       ├── model_file.pth        # Symlink to pretrained model
│       ├── speakers.json         # Our 12,706-embedding d-vector file
│       └── speakers.pth          # Same file, .pth extension
│
├── logs/
│   └── yourtts_finetune_v2.log   # Live training log (tail -f to watch)
│
└── notebooks/
    ├── 01_environment_setup.ipynb
    ├── 02_data_pipeline.ipynb
    ├── 03_whisper_asr_finetuning.ipynb
    ├── 04_speaker_encoder_finetuning.ipynb
    ├── 05_translation_finetuning.ipynb
    ├── 06_emotion_detector.ipynb
    ├── 07_tts_finetuning.ipynb
    └── 08_pipeline_integration_demo.ipynb
```

---

## 18. Datasets Used

### ASR Training (Whisper fine-tuning)
| Dataset | Language | Speakers | Hours |
|---------|----------|----------|-------|
| CommonVoice Telugu | Telugu | ~2,000+ | ~50h |
| FLEURS Telugu | Telugu | ~100 | ~12h |

### Speaker Encoder Training
| Dataset | Description |
|---------|-------------|
| VoxCeleb1/2 | 7,000+ speakers, 1M+ clips (pretrained base) |
| Telugu speaker data | Language-agnostic fine-tuning |

### Emotion Detector Training
| Dataset | Description |
|---------|-------------|
| EmoV-DB | Scripted emotional speech, professional actors, 4 emotions + neutral |
| Labels converted to continuous VAD scores |

### Translation Fine-tuning
| Dataset | Description |
|---------|-------------|
| CoVoST-2 (te-en) | Telugu→English speech translation pairs |
| IndicCorp | Parallel text corpus for Indic languages |

### TTS Fine-tuning
| Dataset | Speakers | Clips | Notes |
|---------|----------|-------|-------|
| LibriTTS dev-clean | 40 | 5,736 | Diverse English text, 16kHz |
| LibriTTS test-clean | 39 | 4,837 | Diverse English text, 16kHz |
| Speech Accent Archive | 2,133 | 2,133 | Same passage, 200 accents, converted from MP3 |
| **Total** | **2,212** | **12,706** | |

---

## 19. Key Design Decisions & Why

### Why YourTTS over XTTS?
XTTS v2 is newer and produces higher quality audio but requires ~4 GB VRAM just for the model. With 8GB total and needing Whisper + NLLB + Emotion all loaded simultaneously, XTTS leaves no room. YourTTS uses ~0.3 GB and still supports zero-shot voice cloning.

### Why NLLB-600M over IndicTrans2?
IndicTrans2 is specifically designed for Indic languages and often outperforms NLLB on Telugu. However, its architecture makes LoRA fine-tuning and cross-attention extraction more complex. NLLB's well-documented API made it easier to hook into for cross-attention matrix extraction needed for prosody alignment.

### Why keep speaker encoder on CPU?
The ECAPA-TDNN speaker encoder is only 23MB and runs in ~0.05 seconds per utterance on CPU. Putting it on GPU would waste ~0.3 GB VRAM that's needed for Whisper. It runs in parallel with GPU inference, so it doesn't add wall-clock time.

### Why Speech Accent Archive for TTS training?
The archive has 2,133 speakers covering 200 native language backgrounds — an extremely diverse set in just 2-3 hours of audio. Each speaker reads the SAME passage ("Please call Stella..."), which means the text content variable is controlled. The model can purely learn speaker diversity without needing to also learn from varied text content.

### Why two-stage voice cloning (YourTTS + FreeVC24)?
YourTTS d-vector conditioning works at the latent code level and is limited by how well the 512-dim vector captures fine-grained voice characteristics. FreeVC24's WavLM-based content-speaker disentanglement operates at a much finer acoustic level. The two stages are complementary: YourTTS gets the voice roughly right, FreeVC24 refines it.

### Why batch_size=1?
VITS uses dual optimizers (generator + discriminator). PyTorch's gradient scaler and the speaker encoder loss both require additional memory buffers per batch. At batch_size=2, peak GPU memory during the optimizer step exceeds 8GB (OOM). At batch_size=1 we use ~1.7GB with plenty of headroom.

---

## 20. Evaluation Metrics

| Metric | What it measures | Target |
|--------|-----------------|--------|
| **SECS** | Speaker encoder cosine similarity between input and output voice | > 0.70 |
| **WER** (Word Error Rate) | ASR accuracy on Telugu | < 15% |
| **BLEU** | Translation quality (Telugu→English n-gram overlap) | > 25 |
| **MOS** | Mean Opinion Score — human rating of audio quality (1–5) | > 3.5 |
| **RTF** | Real-Time Factor — processing time / audio duration | < 2.0 |
| **Latency** | End-to-end time from input to output | < 5s |

### SECS history on this project
- Pretrained YourTTS (6 speakers) + ECAPA-TDNN metric: **0.35–0.38** (both problems present)
- After fixing metric to use YourTTS encoder: measured with same encoder space
- After fine-tuning on 2,212 speakers: **(training in progress — target > 0.70)**

---

## 21. Glossary of Terms

| Term | Meaning |
|------|---------|
| **ASR** | Automatic Speech Recognition — converting speech to text |
| **TTS** | Text-to-Speech — converting text to speech |
| **d-vector** | Discriminative vector — a speaker embedding used for TTS conditioning |
| **Speaker embedding** | A compact numerical representation of a person's voice identity |
| **VITS** | Variational Inference with adversarial learning for Text-to-Speech — the architecture used by YourTTS |
| **HiFi-GAN** | A waveform generation model (vocoder) used as the decoder in VITS |
| **VAD** | Valence-Arousal-Dominance — a 3D continuous model of emotion |
| **FiLM** | Feature-wise Linear Modulation — a conditioning technique using scale+shift |
| **LoRA** | Low-Rank Adaptation — an efficient fine-tuning technique |
| **NLLB** | No Language Left Behind — Facebook's 200-language translation model |
| **ECAPA-TDNN** | A type of speaker encoder architecture using time-delayed neural networks with attention |
| **Wav2Vec2** | A self-supervised speech model from Facebook used as backbone for emotion detection |
| **SECS** | Speaker Encoder Cosine Similarity — metric for voice cloning quality |
| **Zero-shot voice cloning** | Cloning a voice never seen during training, using only a short audio reference |
| **INT8 quantization** | Storing model weights as 8-bit integers (4× memory savings) |
| **NF4 quantization** | 4-bit normal float quantization (8× memory savings) |
| **GAN** | Generative Adversarial Network — generator + discriminator trained competitively |
| **KL divergence** | A measure of how different two probability distributions are (used in VAE training) |
| **Cross-attention** | Attention mechanism where queries come from one sequence and keys/values from another |
| **Mel spectrogram** | Time-frequency representation of audio aligned to human pitch perception |
| **F0** | Fundamental frequency — the pitch of voiced speech in Hz |
| **Prosody** | The rhythm, stress, and intonation patterns of speech |
| **Beam search** | A search algorithm for sequence generation that keeps k best candidates at each step |
| **ResBlock** | Residual block — a neural network block with a skip connection for training stability |

---

*Training currently running: ~200/5,210 steps/epoch. Watch live with `tail -f logs/yourtts_finetune_v2.log`*
