# 调研：new-vc-enh（2026-11-24）

REPORT: newest voice conversion, timbre refiners, vocoders/BWE and speech enhancement as add-ons to the GPT-SoVITS V5 pipeline (as of 2026-11-24)

Network limits in this session: arxiv.org, huggingface.co, *.github.io and modelscope could not be reached. GitHub repos were read directly with git clone and WebFetch. Paper numbers marked [snippet] come from search-engine extracts of the paper and were not read in the paper itself. Anything marked [unverified] is my inference or general knowledge.

## 0. Pipeline facts that change the picture (from /home/user/voicetwin/research/gsv_v4_v5/记录.md and GPT-SoVITS README @6d51ffc, 2026-11-12)
- V5 is a CFM (diffusion) SoVITS with LoRA fine-tuning (`s2_train_v3_lora.py`) plus a new vocoder. The final file is **48 kHz**. The official README says "An updated vocoder significantly reduces high-frequency spectral mirroring and aliasing artifacts." So bandwidth extension or a vocoder swap has almost nothing left to fix.
- GPT-SoVITS ships **no vocoder training code**: only `GPT_SoVITS/BigVGAN/*` inference files, `s2_train*.py`, and no discriminators.
- VoiceTwin's "像你本人 %" scorer uses ERes2NetV2, 3D-Speaker CAM++ and ReDimNet2, all at **16 kHz**. Anything above 8 kHz (BWE, vocoder "air") is invisible to it, so those gains can only be judged by listening.

## 1. Direct evidence that TTS -> VC raises similarity: weak, and it points against a fine-tuned TTS
- **Voice Filter** (Amazon, ICASSP 2022, arXiv 2202.08164) [snippet]:
  - Setup: single-speaker TTS followed by VC as a post-processing module, using 1 minute of target data.
  - MUSHRA speaker similarity: Voice Filter **70.38±0.93**, versus a fine-tuned TTS (TTS-DA, 30× more data) **73.67±0.85**. Reference recordings scored 83.23.
  - The authors themselves report a "4% relative degradation in speaker similarity".
  - Conclusion: the VC cascade substitutes for fine-tuning when data is scarce; it did not beat a fine-tuned TTS.
- **arXiv 2610.29351** (2026, cross-lingual accent control) [snippet]:
  - Their method beat a TTS->VC cascade on speaker similarity: ECAPA +0.091 and CAM++-EN +0.094.
  - The cascade had higher DNSMOS but **more word errors**. This matters for the English sentences.
- **ADD 2022** time-domain post-processing (arXiv 2204.08692) [snippet]: speaker cosine similarity fell from 0.91 to 0.88 after post-processing.
- I found **no published A/B of GPT-SoVITS -> RVC or Seed-VC.** Hobby "TTS + RVC" pipelines exist, but with no numbers.
- Bottom line: with 2.5 h of data and a fine-tuned V5, a VC stage is **not proven to help**. Only a measured, held-out A/B can decide.

## 2. Voice conversion candidates

**Seed-VC** (github.com/Plachtaa/seed-vc)
- Releases: v1 in 2024, v2 (AR + CFM, voice/accent conversion) in 2025. The repo was **archived on 2025-11-21**.
- Licence: **GPL-3.0**, which is incompatible with bundling into VoiceTwin (MIT). It would have to run as a separate tool.
- Models:
  - 22.05 kHz speech models (v1 whisper-small, v2).
  - **44.1 kHz f0-conditioned model** `config_dit_mel_seed_uvit_whisper_base_f0_44k.yml` (200M parameters). This model can keep V5's own pitch contour and Chinese tones.
- Fine-tuning: README says "minimum 1 utterance per speaker … minimum 100 steps, 2 min on T4". Files must be 1–30 s. README also says fine-tuning "will largely improve speaker similarity on particular speakers, but may slightly increase WER" (no numbers given).
- Published numbers (EVAL.md; SECS measured with Resemblyzer, which is a weak model):
  - Zero-shot on LibriTTS: Seed-VC **0.8676** SECS / WER 11.99 / CER 2.92. CosyVoice 0.8440 / 18.98. OpenVoice 0.7547 / 15.46. Ground truth WER 8.02 / CER 1.57, so VC itself **adds errors**.
  - Against per-speaker So-VITS-4.0: higher SECS for all 3 speakers (e.g. 0.8899 vs 0.8637).
  - Singing, against per-speaker RVCv2: SECS 0.7405 vs 0.7264 and CER 19.70 vs 28.46, but DNSMOS OVRL 3.06 vs 3.12. EVAL.md says Seed-VC's audio quality is slightly lower than RVCv2's.
- Speed: RTX 3060 laptop, tiny model, 150 ms per 0.18 s block. No VRAM figures published.
- Windows: supported. `requirements.txt` pins `torch==2.4.0`; RTX 50 needs a cu128 build of torch 2.7 or later [unverified that it runs in the GPT-SoVITS env].
- **Metric-hacking risk (checked in the code):** every v1 config conditions on `campplus_cn_common.bin`, which is the same CAM++ model as one of VoiceTwin's scorers.

**RVC via Applio** (github.com/IAHispano/Applio)
- Release: 3.7.2 on 2026-11-19. Upstream RVC is stable rather than new. Applio says it now gets "no frequent updates".
- Licence: **MIT** for code and weights; commercial use allowed.
- Training configs at 32k, 40k and **48k**. Vocoders: HiFi-GAN-NSF, MRF, RefineGAN.
- RMVPE f0 is taken from the source audio, so V5's intonation and tones are kept. Output is frame-synchronous, so timing does not change [general RVC behaviour, unverified in Applio code]. The feature-retrieval index draws on her real recordings.
- Windows: `run-install.bat` installs **cu128** wheels, so RTX 50 is supported (checked in the repo).
- No published speech similarity, MOS or WER numbers. The only table is the singing comparison above (SECS 0.7264, DNSMOS OVRL 3.12).

**MeanVC / MeanVC2** (github.com/ASLP-lab/MeanVC; arXiv 2510.08392 and 2606.09050)
- Licence: Apache-2.0. MeanVC2 has 18M parameters and 110 ms latency [snippet].
- Inference writes **16 kHz** (`infer.py`). It conditions on a WavLM-large speaker-verification model.
- Built for real-time streaming. Output would be a bandwidth downgrade from 48 kHz. **Reject.**

**X-VC** (github.com/Jerrister/X-VC; arXiv 2604.12456, April 2026)
- Licence: MIT. Config is `sample_rate: 16000`. It uses an **ERes2Net** speaker encoder, the same family as VoiceTwin's scorer.
- Seed-TTS-eval WER 1.99 zh / 2.83 en [snippet from an aggregator]. No SIM table seen.
- **Reject:** 16 kHz output and scorer overlap.

**Amphion Vevo / Vevo1.5 / Vevo2** (Vevo2 released 2026-03-25)
- Code is MIT, but the weights are trained on Emilia. HF cards: Vevo **CC-BY-NC-4.0**, Vevo1.5 **CC-BY-NC-ND-4.0** [snippet]. Vevo2 weights licence: unverified.
- Vevo2 uses a Vocos-based vocoder. The "style-converted" modes regenerate prosody with an AR model, which risks content errors.
- **Reject:** non-commercial licence, and AR regeneration of prosody.

**CosyVoice VC**
- Licence: Apache-2.0. `inference_vc` exists only in the v1 class, and that model runs at 22.05 kHz (v2/v3 configs are 24 kHz).
- No VC numbers beyond the Seed-VC table (SECS 0.8440, WER 18.98).
- Linux-only bits: the `ttsfrd` wheel is a Linux x86_64 build. **Reject.**

**Chatterbox VC** (Resemble AI)
- Licence: MIT. Marked "retired research" as of Aug 2026. Runs at 24 kHz.
- `vc.py` **always applies the Perth watermark** (line 103). This alters the signal and risks the digital-silence rule. **Reject.**

**OpenVoice V2**
- Licence: MIT; last push 2025-04. Lowest SECS in the Seed-VC table (0.7547). **Reject.**

**kNN-VC / kNN-TTS** (NAACL 2025)
- Retrieval over WavLM frames at 16 kHz. kNN-TTS had the best objective SIM, but listeners rated HierSpeech++ higher [snippet]. **Reject** for the 16 kHz output.

## 3. Vocoders and bandwidth extension
- **BigVGAN-v2** (MIT): final checkpoints Sep 2024, maximum 44 kHz. **Vocos** (MIT): 2023. Both are older than the V5 vocoder (Oct 2026). Swapping in either needs a matching mel configuration, and no evidence shows it helps.
- **AP-BWE** (MIT; supports 8/12/16/24 -> 48 kHz) is already in GPT-SoVITS as `tools/AP_BWE_main/24kto48k`, used for v3's 24 kHz output.
- **ClearerVoice MossFormer2_SR_48K** (Apache-2.0; fine-tuning script added 2025.4) does the same job.
- These are useful only if a 22.05/24 kHz VC output had to be brought back to 48 kHz. V5 is already 48 kHz.
- Speaker-adaptive vocoder fine-tuning: older evidence says it helps (2018 work: MOS 3.80/3.77 with 10 minutes of data; 2021 MelGAN "adapt90" result) [snippets]. But GPT-SoVITS has no vocoder or discriminator training code, so this would be a from-scratch build. **Not recommended now.**

## 4. Speech enhancement and restoration
- **Sidon** (MIT code; sidon-v0.1 checkpoint MIT; w2v-BERT 2.0 + HiFi-GAN at 48 kHz): about Miipher quality, and TTS trained on cleaned data improved [snippet].
- **ClearerVoice SE 48k** (Apache-2.0). **Resemble Enhance** (MIT; last push 2024-12).
- All generative or resynthesis enhancers can **shift timbre or content**. TIMUSE (2610.34814) and PASE (AAAI 2026) say so explicitly and now report RawNet3 speaker similarity and WER for that reason [snippets].
- V5 output is already clean. Enhancing her training data would make it less like her real recordings, which are the 100 % anchor. VoiceTwin already runs Demucs and denoising on training data.
- **Not recommended as a post-step.** At most, test it on noisy training segments only.

## 5. Risks for any post-step
- **Absolute silence:** every VC or vocoder emits low-level noise in pauses. The VC must run per sentence before VoiceTwin's trim and digital-silence step, and the silence checks must run again afterwards.
- **Pronunciation:** VC raises WER/CER (Seed-VC zero-shot CER 1.57 -> 2.92; the 2026 cascade had more word errors). The existing ASR check must gate every VC candidate, especially the English words.
- **Metric hacking and leakage:**
  - Seed-VC uses CAM++ internally, so score its outputs without CAM++.
  - RVC's index and Seed-VC fine-tuning use her recordings, so compute "像你本人 %" against **held-out** recordings used neither in V5 training nor in VC training.
- **Bandwidth:** never accept a 16/22/24 kHz path into a 48 kHz product.
- **Licence:** Seed-VC is GPL-3.0, and the repo is archived (no fixes).

## 6. Recommendation (at most 2; both optional and measured, none proven)
Integrate as **extra candidates inside the existing per-sentence picker**, never as a forced stage. Each V5 candidate also gets a VC-processed twin. The picker chooses, so raw V5 still wins when the VC is worse. That makes a regression impossible as long as the scorer is unbiased.

1. **RVC 48k via Applio 3.7.x**
   - Why: MIT, Windows/cu128 ready, f0 taken from V5 (pitch shift 0), timing preserved, retrieval from her own recordings.
   - Setup: train on the same 2.5 h minus a held-out set.
   - Things to tune: index rate, `protect`.
2. **Seed-VC v1 44.1 kHz f0-conditioned, fine-tuned on her data**
   - Why: strongest published open VC similarity, and README claims fine-tuning "largely improve[s] speaker similarity".
   - Conditions: run as a separate GPL tool; resample 44.1 -> 48 kHz; leave CAM++ out of scoring for these candidates; torch for RTX 50 needs checking first.

**Acceptance test (one round):**
- Test material: about 50 zh and 20 en/mixed sentences. Compare V5 raw vs V5->RVC vs V5->Seed-VC.
- Metrics:
  - Held-out SV score using ReDimNet2 and ECAPA.
  - CER/WER from the existing ASR.
  - F0 RMSE against V5 raw.
  - The digital-silence check after re-gating.
  - Blind ABX by the teacher (the only test of "can't tell").
- Keep a tool only if held-out SV rises **and** CER does not rise **and** ABX does not prefer raw V5. Otherwise drop it after that one round.

Not recommended: zero-shot VC at 16–24 kHz (MeanVC/MeanVC2, X-VC, CosyVoice, Chatterbox, OpenVoice, kNN-VC), NC-licensed Vevo, vocoder swaps, BWE, and post-TTS enhancement.

## Sources
- https://github.com/Plachtaa/seed-vc (README, EVAL.md, configs/presets/*.yml showing campplus_path, requirements.txt)
- https://github.com/IAHispano/Applio (README, LICENSE, run-install.bat cu128, rvc/configs/48000.json) and https://github.com/IAHispano/Applio/releases
- https://github.com/ASLP-lab/MeanVC (src/infer/infer.py at 16000 Hz); MeanVC2 arXiv 2606.09050 [snippet]
- https://github.com/Jerrister/X-VC (configs/xvc.yaml at 16000 Hz, ERes2Net); arXiv 2604.12456 [snippet]
- https://github.com/open-mmlab/Amphion (Vevo2 2026/03/25; models/svc/vevo2/README.md); HF licences for amphion/Vevo and Vevo1.5 [snippet]
- https://github.com/FunAudioLLM/CosyVoice (cosyvoice/cli/cosyvoice.py inference_vc; examples/libritts/*/conf/*.yaml sample rates)
- https://github.com/resemble-ai/chatterbox (src/chatterbox/vc.py Perth watermark; s3gen/const.py S3GEN_SR=24000)
- https://github.com/myshell-ai/OpenVoice; https://github.com/bshall/knn-vc; kNN-TTS https://aclanthology.org/2025.naacl-short.65.pdf [snippet]
- https://github.com/RVC-Boss/GPT-SoVITS (README V5 Release Notes; tools/AP_BWE_main; no vocoder training code)
- https://github.com/NVIDIA/BigVGAN; https://github.com/gemelo-ai/vocos; https://github.com/yxlu-0102/AP-BWE
- https://github.com/modelscope/ClearerVoice-Studio; https://github.com/sarulab-speech/Sidon (arXiv 2509.17052 [snippet]); https://github.com/resemble-ai/resemble-enhance
- Voice Filter: https://arxiv.org/pdf/2202.08164 [snippet]; arXiv 2610.29351 [snippet]; arXiv 2204.08692 [snippet]; arXiv 2610.34814 and PASE (AAAI 2026) [snippets]

Local files (scratchpad, shallow clones used for verification): /tmp/claude-0/-home-user/3d87cc6c-41dd-58c2-802f-d51dc0c78527/scratchpad/repos/. No files under /home/user/voicetwin were changed.
