# 调研：new-tts（2026-11-24）

# Newest open-source TTS and voice-cloning models (to Nov 2026) as helpers for GPT-SoVITS V5, for the teacher's voice

## Context that changes the ranking (from repo docs, read-only)
- **Teacher's GPU:** RTX 5070 Laptop with **8 GB VRAM** (about 6.8 GB free). Source: /home/user/voicetwin/教学手册.md line 188. Anything above roughly 7 GB at inference, and almost any 2B+ fine-tune, will not run locally.
- **Engines already in the app:** Qwen3-TTS and IndexTTS 2.5 are already wired in as zero-shot backup engines (教学手册.md line 63). The real new options are the ones below.
- **Known problem with mixing engines:** mixing engines or sample rates sentence by sentence makes the audio sound "brighter, then duller" (research/一模一样/记录.md line 106). Any helper engine with a different bandwidth (24 kHz against 48 kHz) should be chosen per lecture or used only as a benchmark, not mixed per sentence.
- **V5 facts:**
  - The V5 release commit is dated 2026-10-05, alongside the CUDA Graph + Flash Attention commit (https://raw.githubusercontent.com/RVC-Boss/GPT-SoVITS/main/docs/en/Changelog_EN.md).
  - Official zero-shot Seed-TTS **test-zh** SIM: v4 0.735, v5dev 0.805, v5pro 0.824 (v5pro not released), GT 0.755. These were read off the README chart image; the transcription is in /home/user/voicetwin/research/gsv_v4_v5/记录.md.
  - No official numbers exist for test-en SIM or for V5 after fine-tuning.

## Caveat on every SIM number below
- The figures are zero-shot Seed-TTS-eval SIM (WavLM-SV cosine similarity to the prompt clip).
- The real-speaker ground truth scores only zh 0.755 and en 0.734 (LongCat table). Models scoring above 0.8 are copying the prompt's recording conditions; that does not mean listeners can't tell.
- These figures do not predict how a model compares after fine-tuning. Only the app's own calibrated "像你本人 %" score on held-out real sentences, plus a blind listening (ABX) test, can decide.
- The only human "can't tell" metric I found: Fish S2-Pro Audio Turing Test 0.515 posterior mean, against 0.417 for Seed-TTS. It is self-reported (https://github.com/fishaudio/fish-speech), and I have not verified what the scale means.

## Candidates (zero-shot Seed-TTS-eval SIM: zh / en; licence; fine-tuning; VRAM; Windows)

### Highest published speaker similarity
- **LongCat-AudioDiT 3.5B / 1B** (Meituan)
  - Release: arXiv 2603.29339 (submitted 31 Mar 2026); repo update 2026.10.15 fixed volume normalisation in load_audio.
  - SIM zh / en: **0.818 / 0.786** for 3.5B and 0.812 / 0.762 for 1B. Highest published zh and en SIM I found; above Seed-DiT 0.809 / 0.790.
  - Other: zh-hard 0.797; CER 1.09. **MIT** licence for code and weights. Diffusion TTS on waveform latents, 24 kHz output. No fine-tuning code.
  - VRAM (third-party, unverified):
    - 3.5B: ~9 GB bf16, ~8 GB fp8, 15.06 GiB peak at full precision.
    - 1B: ~4 GB bf16, 5.78 GiB peak at full precision.
    - RTF: 0.13 (1B), 0.28 (3.5B).
    - Sources: TTS-Q arXiv 2610.12474; ayousanz.hatenadiary.jp.
  - Windows: not stated.
  - https://github.com/meituan-longcat/LongCat-AudioDiT
- **dots.tts** (rednote-hilab)
  - Release: 2026.06 (report arXiv 2606.07080). Inference speed-up 2026.07; **dots.tts.edit** 2026.08; edit v1.1 2026.10; doteBench 2026.09.
  - Licence: Apache-2.0 for code and checkpoints ("research and authorized deployment").
  - Model: 2B, 48 kHz, 24 languages.
  - SIM zh / en / zh-hard:
    - **SOAR:** **81.0 / 77.1** / 79.5; WER zh 0.94, en 1.30.
    - **MF (4-step):** 80.0 / 76.2.
  - Other benchmarks: CV3 cross-lingual SIM en→zh 75.0 and zh→en 72.8 (CosyVoice3 66.9 / 66.4).
  - Fine-tuning: script provided (`scripts/train_dots_tts.py`, JSONL of fid/audio/text). No hardware, LoRA or single-speaker guidance.
  - VRAM (measured on H800, ≤256 patches): SOAR peak 7.86 GB, MF 5.74 GB. RTF 0.20 / 0.15.
  - Windows: not mentioned. Third parties say Linux/macOS only (WSL2) — unverified.
  - dots.tts.edit supports `<sub>/<ins>/<del>/<pitch>/<rate>/<pause/>` editing inside existing audio (zh+en). No published scores.
  - https://github.com/rednote-hilab/dots.tts
- **FireRedTTS3** (FireRedTeam)
  - Release: Base 2026-08-05 on HF (seen via search); code and Instruct 2026-08-13.
  - SIM zh / en / zh-hard: **80.9 / 77.2** / 78.4; WER en 1.64.
  - Licence: Apache-2.0 LICENSE file, but the README says zero-shot cloning is "solely for academic research purposes".
  - Size, VRAM, fine-tuning and Windows: none stated. Has instruction-based voice design and speech editing.
  - https://github.com/FireRedTeam/FireRedTTS3
- **VoxCPM2** (OpenBMB)
  - Release: 2026.04.
  - SIM zh / en: **79.5 / 75.3**.
  - Licence: Apache-2.0, "free for commercial use". 2B, 30 languages, 48 kHz.
  - **Official LoRA and full fine-tuning**, with a LoRA web UI:
    - Docs: 5–50 clips for one speaker, 1–3 epochs, overfitting can start within a few hundred steps.
    - VRAM: **~20 GB for LoRA, ~40 GB for full fine-tuning at batch size 16** (voxcpm.readthedocs.io, seen via search; site blocked here).
    - The older VoxCPM1.5 (2025.12) also has SFT and LoRA.
  - Inference: ~8 GB VRAM, RTF 0.30 on a 4090.
  - Windows: not mentioned officially; community Windows bundles exist (CSDN, third-party).
  - https://github.com/OpenBMB/VoxCPM
- **MOSS-TTS** (OpenMOSS)
  - Releases: v1.0 2026-02-10; v1.5 2026-05-26; Local-Transformer-v1.5 (4B, 48 kHz) 2026-06-18.
  - SIM zh / en: MossTTSLocal 1.7B **79.62 / 73.28**; 8B Delay 76.98 / 70.86.
  - Licence: Apache. Fine-tuning tutorials provided.
  - 8B models need llama.cpp or a TRT 8 GB config to fit.
  - https://github.com/OpenMOSS/MOSS-TTS

### Mid-tier
- **Ming-3.0-tts** (inclusionAI)
  - Release: 2026.10. Licence: MIT.
  - SIM zh / en: 0.5B **0.758 / 0.691**; "tiny" MoE (7.9B total, 1.3B active) 0.781 / 0.728.
  - Fine-tuning: badge says "0.5B Full/LoRA", but the roadmap still lists fine-tuning as unchecked. Tested on H800/H20 only.
  - https://github.com/inclusionAI/Ming-3.0-tts
- **CosyVoice**
  - Fun-CosyVoice3-0.5B-2512 (2025-12): zh / en 78.0 / 71.8. Apache. Training recipes exist; Linux-only (ttsfrd wheel). No open 2026 checkpoint.
  - **CosyVoice 3.5** is API-only on Alibaba Model Studio; zh 0.797 / en 0.738 per the LongCat table, which takes them from a WeChat article.
  - https://github.com/FunAudioLLM/CosyVoice
- **Qwen3-TTS** (already in the app)
  - Release: 2026-01-22. Apache.
  - SIM zh / en: 0.770 / 0.717 (LongCat table).
  - **Single-speaker SFT script** in `finetuning/` (1.7B-Base): same ref_audio for every sample; examples use bs 32 / lr 2e-6 / 10 epochs or bs 2 / 2e-5 / 3 epochs. No VRAM or data-size guidance.
  - https://github.com/QwenLM/Qwen3-TTS/tree/main/finetuning
- **IndexTTS-2.5** (already in the app)
  - Release: 2026-08-10. bilibili Model Use License (commercial use by contacting bilibili).
  - Benchmarks: only CV3-Eval average WER / SS 6.75 / 73.18 (RL version 6.00 / 73.63); no Seed-TTS SIM.
  - RTF 0.21 on a 4090. Covers Windows. No fine-tuning.
  - https://github.com/index-tts/index-tts
- **GLM-TTS** (Zhipu)
  - Release: 2025-12-11. Apache.
  - SIM zh: 76.1 (RL version 76.4).
  - LoRA config and GRPO script in the repo. The report claims LoRA reaches full fine-tuning quality with **1 h of single-speaker audio**; I did not verify this in code.
  - https://github.com/zai-org/GLM-TTS

### Not realistic for this teacher
- **Fish Audio Drama 3** (~2026-10-13): 35B-A3B; Fish Audio Community License with a revenue limit; reportedly 48 GB GPU (secondary sources). S2-Pro (Mar 2026) is 4B under a research licence.
- **Higgs Audio v3:** 2026-06-04, 4B, non-commercial licence.
- **Breeze TTS 2:** 2026-08-25, zh+en, non-commercial weights; its claim of #1 among open weights is a vendor claim.
- **Raon-OpenTTS:** trained on English-only data, CC-BY-NC.
- **MegaTTS3:** WaveVAE encoder withheld, so no free cloning (zh / en 79.0 / 77.1).
- **Step-Audio-EditX:** Apache, ~12 GB VRAM, editing and post-processing; last update 2026-01-29.
- **Seed-VC:** **archived 2025-11-21**, GPL-3.0 (https://github.com/Plachtaa/seed-vc).
- **OmniVoice** (Xiaomi, 0.6B, 600+ languages): weights licence unclear; tts-bench says CC-BY-NC (unverified); no zh SIM published.
- **Older context:** F5-TTS (last release 1.1.22 on 2026-07-23, non-commercial weights, zh 0.741), Spark-TTS 0.672, MaskGCT 0.774, VibeVoice 74.4.
- **Unverified, no primary source found:** VoxMel, Audio8 TTS Realistic, Reverb-TTS. The only source is https://github.com/5uck1ess/tts-bench.
- **Not checked:** Kimi-Audio, Llasa.
- **No GPT-SoVITS successor or fork** beats V5; official v5pro is unreleased.

## Top 3 to assist or compare against V5
1. **dots.tts (SOAR or MF), plus dots.tts.edit**
   - Why:
     - Best Apache-licensed zh / en SIM (81.0 / 77.1).
     - 48 kHz like the V4 route.
     - MF fits in about 6 GB.
     - Has a fine-tuning script.
     - The edit model can swap one mis-said word, especially an English one, inside an otherwise good V5 sentence instead of regenerating the whole sentence.
   - How to plug in:
     - Extra per-sentence candidates, scored by the existing "像你本人 %" / ASR / rate / pitch scorer.
     - Second opinion for full-English sentences.
     - Word repair: check that the unedited audio is left unchanged (unverified).
   - Risks: Windows unsupported or untested (would need WSL2 or a separate process); no single-speaker fine-tuning guide.
2. **VoxCPM2**
   - Why: the only top-tier (79.5 / 75.3) Apache, 48 kHz model with an official, documented single-speaker LoRA path. This allows a fair fine-tuned vs fine-tuned comparison with V5 on held-out recordings.
   - Limit: LoRA officially needs ~20 GB at batch size 16, so the 8 GB laptop is unverified (it might work at batch size 1).
   - Option: train on a GitHub Actions GPU larger runner (Tesla T4, 16 GB; Team or Enterprise Cloud plan only; https://docs.github.com/en/actions/reference/runners/larger-runners), then run inference on the laptop (~8 GB, borderline).
3. **LongCat-AudioDiT (1B locally, 3.5B in fp8 or on a cloud GPU)**
   - Why: highest published zh / en SIM (0.818 / 0.786) and MIT licence.
   - Use it as a zero-shot ceiling check: if fine-tuned V5 doesn't beat it on held-out sentences, the training needs work. Also as a second opinion for English.
   - Limits: 24 kHz output means don't mix it per sentence with 48 kHz V5; no fine-tuning.

FireRedTTS3 has stronger numbers than VoxCPM2 (80.9 / 77.2), but I ranked it lower because its README limits cloning to academic research and it publishes no size, VRAM or fine-tuning details.

## Blocked in this session
huggingface.co, arxiv.org, alphaxiv, readthedocs, artificialanalysis.ai, marktechpost and ai-tldr.dev did not resolve here. Numbers attributed to those sites came from search snippets and are secondary. Everything sourced from GitHub READMEs was fetched directly. I did not edit any file under /home/user/voicetwin.
