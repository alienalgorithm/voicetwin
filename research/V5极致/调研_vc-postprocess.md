# 调研：vc-postprocess（2026-11-24）

## VC and enhancement post-pass after V5: research report (read-only; no files in /home/user/voicetwin were changed)

### Recommendation
- **Do not add a voice-conversion (VC) or enhancement step to the default pipeline.** I found no controlled evidence, from a paper, benchmark or official GPT-SoVITS source, that a VC pass on top of GPT-SoVITS output makes it sound more like the speaker. Every candidate I checked also brings a concrete risk.
- **At most, one optional experiment, and only if the teacher's real V5 training still shows a timbre gap:** RVC v2 at 48 kHz.
  - It is the only candidate that keeps all four requirements at once: 48 kHz output, MIT licence, an RTX 50 install path and fine-tuning on her data.
  - Add it as a third whole-document version through the existing A/B mechanism. Keep it only if it passes the gates in section 4, including her blind listening.
- **VC cannot improve tone (语气).** RVC takes the pitch contour from its input (RMVPE f0) and keeps the timing, so tone, rhythm and speaking habits stay exactly what V5's GPT stage produced. This is my judgement from how RVC works. Only Seed-VC V2's `--convert-style` claims to convert accent and emotion, and section 2 rejects it.
- **Super-resolution and enhancement add nothing to V5** (section 3).

### 1. Evidence on "TTS then VC" improving speaker similarity
- **No controlled study found.** Several searches returned no study measuring GPT-SoVITS (or any strong fine-tuned TTS) with and without a VC pass.
  - The only support is anecdotal: one practitioner's log reports a "drastic increase in quality" for XTTS followed by RVC (https://tahadidit.github.io/python-tts-logs).
  - Community UIs such as XTTS-RVC-UI (https://github.com/rosenio/XTTS-RVC-UI) and AllTalk also chain TTS into RVC.
  - XTTS is zero-shot and much weaker than a fine-tuned V5, so this does not transfer to our case.
- **V5 already does what a diffusion VC does.**
  - V5's SoVITS generates a mel spectrogram by CFM sampling, conditioned on reference features and the reference mel (`GPT_SoVITS/module/models_v5.py`: `CFMV5`, `synthesize_v5_mel` L51-78; `TTS_infer_pack/TTS.py` L1812). A 48 kHz vocoder then makes the waveform (TTS.py L726-759: `sr` 48000).
  - Seed-VC is the same family: a diffusion transformer conditioned on the whole reference clip.
  - Judgement: a second model of the same kind, trained on the same data, has little room to add timbre, and its vocoder is worse (22.05 kHz).
- **VC adds recognition errors.** In Seed-VC's own EVAL.md (LibriTTS), WER goes from 8.02 on the ground truth to 11.99 after Seed-VC, 15.46 after OpenVoice and 18.98 after CosyVoice. CER goes from 1.57 to 2.92, 4.73 and 7.29. The Seed-VC README (L163) says fine-tuning "will largely improve speaker similarity on particular speakers, but may slightly increase WER", without numbers. So the ASR check would have to run again after any VC pass.
- **Our automatic score cannot be the only judge.**
  - VoxSim (Interspeech 2024, arXiv 2407.18505; numbers from search snippets): ECAPA similarity vs human ratings gives LCC 0.768 on VoxCeleb pairs, and the authors' similarity predictor gives 0.574 on VCC2018 voice-conversion data.
  - YourTTS (arXiv 2112.02418) reports Sim-MOS and SECS disagreeing in one experiment.
  - Seed-VC conditions on a CAM++ speaker embedding (`inference.py` L86-94, `funasr/campplus`). CAM++ is from the same 3D-Speaker family as two of our three scorers (ERes2NetV2, ERes2Net-base). Judgement: it could raise our score without sounding more like her. This is the "自己骗自己" trap already rejected in `research/一模一样/记录.md` L104.
  - Our scoring runs at 16 kHz (`voicetwin/eval/speaker.py` L90/L109), so a loss of bandwidth is invisible to it. The record already notes this for 48k vs 32k (`记录.md` L106).

### 2. Candidates (licences read from each repo's LICENSE on GitHub)

| Tool | Licence / status | Output rate | Fine-tune on her data | RTX 50 (CUDA 12.8) | Verdict |
|---|---|---|---|---|---|
| **RVC** (RVC-Project WebUI) | MIT. Base model trained on ~50 h VCTK, "无版权方面的顾虑" (README L21) | 32/40/48k (v2) | Yes. FAQ: 10-50 min recommended; with clean, long data "200 [epochs] 是ok的" | README L87-102: torch 2.7.1+cu128, Python 3.12, the same torch as the V5 package. But it pins gradio <3.15, pydantic <2 and fastapi <0.100, which conflict with V5's gradio 4.44.1 / pydantic 2.10.6 / fastapi 0.115.2, so it needs its own runtime or an inference-only subset | Only candidate worth one experiment |
| Applio (RVC fork) | MIT plus its own Terms of Use; called actively maintained by the so-vits-svc-fork README | as RVC | yes | gradio 6.20 (separate env) | Fallback if RVC packaging fails |
| **Seed-VC** | GPL-3.0; repo archived 21 Nov 2025 | Speech models 22050 Hz (README L42-45); 44.1k only for the f0-conditioned singing model | Yes: "minimum 1 utterance… 100 steps, 2 min on T4" (README L10) | Pins torch 2.4.0 plus nightly cu126 (requirements.txt L1-7); no cu128 build | Reject: halves V5's bandwidth, unmaintained, scorer-coupling risk, adds WER |
| so-vits-svc | AGPL-3.0; archived 11 Nov 2023; "not intended for production", "avoid… real individuals" | 44.1k, singing-focused | yes | no | Reject |
| so-vits-svc-fork | MIT; README: "No Longer Maintained" | | | | Reject |
| OpenVoice V2 | MIT, free for commercial use | 22050 Hz | No (zero-shot) | n/a | Reject: lowest SECS in Seed-VC's table (0.7547 vs Seed-VC 0.8676 and CosyVoice 0.8440, Resemblyzer) |
| MeanVC / MeanVC2 (2025-26) | Apache-2.0 | 16 kHz vocoder (MeanVC2 README L214); built for streaming | MeanVC2 yes | untested | Reject (16 kHz) |
| Vevo / Vevo1.5 / Vevo2 (Amphion) | Code MIT; weights CC-BY-NC-4.0 (Vevo) and CC-BY-NC-ND-4.0 (Vevo1.5) per HF tags in search snippets | | | | Reject (non-commercial weights; her Bilibili use may count as commercial) |
| Chatterbox VC | MIT; "retired research… no longer maintained" since Aug 2026; every output carries a Perth watermark | | | | Reject |

### 3. Super-resolution and enhancement
- **AP-BWE (MIT): not applicable to V5.**
  - GPT-SoVITS uses it only for v3: TTS.py L1589 and L1653 pass `super_sampling if … version == "v3" else False`. V5's vocoder is natively 48 kHz.
  - Keep `tools/AP_BWE_main` in the package anyway: TTS.py imports it at load time (L39), and leaving it out broke the real run (`research/gsv_v4_v5/记录.md` L112, `datasets1`).
- **resemble-enhance (MIT): reject.**
  - Trained on 44.1 kHz data (README), so it would lower V5's 48 kHz.
  - It is a generative CFM enhancer. The PASE paper (arXiv 2511.13300, search snippet) names "acoustic hallucination: inconsistent speaker characteristics" as a failure mode of generative enhancers. UniPASE (2604.14606) cites a purely generative URGENT 2025 system with the best perceptual scores but large drops in speaker similarity.
- **ClearerVoice-Studio (Apache-2.0).**
  - MossFormer2_SR_48K upsamples input with an effective rate of at least 16 kHz to 48 kHz. V5 is already 48 kHz, so there is no use for it.
  - MossFormer2_SE_48K could replace noisereduce in the "去杂音" B version, only if V5 output shows measurable noise. That version runs only for sentences above `DENOISE_FLOOR_DB` −60 dBFS (`engine.py` L506/L2299). Low priority; untested.

### 4. If the RVC experiment is run (all untested; values are my starting points unless sourced)
1. **Data.**
   - Train on the same confirmed clips as V5, **minus the ~20 held-out validation sentences**. Otherwise the scoring leaks.
   - Use RVC v2 48k with f0 on and RMVPE.
   - Save checkpoints every ~20 epochs up to ~200 (FAQ Q9: 200 is fine for clean data) and pick one with the held-out scoring.
2. **Ceiling test first (cheap, decisive).**
   - Run RVC on her *real* held-out recordings and score them with the existing 「像你本人 %」 (100% = her own held-out level). Also run CER, measure 16-24 kHz band energy and check the pause noise floor.
   - If converting her own voice scores clearly below 100%, or she can hear a difference, stop: VC can only cap similarity.
3. **Main test.**
   - Take the same held-out texts and the V5 best candidates. Compare raw against RVC-converted with paired per-sentence scores: SV %, CER from a new ASR pass, F0 statistics, bandwidth and noise floor.
   - Then a blind ABX with her. The final call is hers, because of the SV-vs-human limits in section 1.
4. **Integration point.**
   - Make it a third whole-document version next to the existing 未去杂音/去杂音 versions (`engine.py` ~L2278-2340: same text, pauses and subtitles, same `_version_score`).
   - Never mix VC and raw sentences within one document. The bandwidth/texture switch is the 忽亮忽暗 problem (`记录.md` L106).
   - Apply VC per sentence **before** `_trim`/`fade` (L1061/L2840) and the stitching with digital silence, so the silence between sentences stays exact.
   - RVC keeps the length, so subtitles stay valid.
   - Run it as a separate process, like the other engines (JSON-lines protocol).
5. **Parameters (from the official parameter descriptions in `i18n/locale/zh_CN.json`).**
   - Transpose 0.
   - `protect` below 0.5 ("保护清辅音和呼吸声，防止电音撕裂等artifact，拉满0.5不开启", L84). This is an official admission that RVC produces such artifacts.
   - `rms_mix_rate` near 0 to keep V5's loudness envelope (L257: closer to 1 means the output envelope is used).
   - `index_rate` 0.5-1 (FAQ Q11: at 1 there is no source-timbre leakage; with high-quality, long data the index matters little).
6. **Cost and risks.**
   - The 50-series package `RVC20240604Nvidia50x0.7z` is reported at ~6.04 GB, sha256 `e5750ddd…bdd52`. This comes from a search result: the original lj1995 upload appears to be deleted and only third-party mirrors remain, so verify before use. Alternatively pip-install the inference dependencies into a separate runtime.
   - On her 8 GB card (RTX 5070 Laptop, `research/开发记录.md` L81), run VC after the GPT-SoVITS API is stopped, to avoid running out of memory.
   - Training time on ~2.5 h of data: unknown, measure on her PC.
   - Other risks: RMVPE octave errors that become pitch glitches, 电音 on consonants and breaths, a noise floor in pauses inside sentences (caught by the existing −60 dBFS check), and new mistakes in the words (re-run ASR).
7. **GitHub Actions (CPU only) can check the plumbing, not the quality.** It can run RVC inference on CPU (README: `requirments_cpu_py312.txt`) with the pretrained v2 48k base and no fine-tuning, on espeak audio. It can assert that the output length and 48 kHz rate are unchanged, that the stitched gaps are exact zeros, and that a crash falls back to the raw version. Quality only shows on her PC.

### 5. What to tell the teacher honestly
- No published proof exists that converting V5's output with another model makes it sound more like her.
- It cannot change tone or rhythm.
- It can add new mistakes in the words and electronic-sounding artifacts.
- So it is not in the standard flow. If the real V5 still sounds unlike her in timbre, the program will try RVC once on her own recordings first, and she decides by blind listening.

### Sources
- **GPT-SoVITS (main):**
  - https://github.com/RVC-Boss/GPT-SoVITS/blob/main/GPT_SoVITS/TTS_infer_pack/TTS.py (L39, L726-759, L1589, L1653, L1755-1767, L1812)
  - https://github.com/RVC-Boss/GPT-SoVITS/blob/main/GPT_SoVITS/module/models_v5.py
  - https://github.com/RVC-Boss/GPT-SoVITS/blob/main/tools/audio_sr.py
- **Seed-VC:**
  - https://github.com/Plachtaa/seed-vc (README L10, L42-45, L88, L163; requirements.txt; inference.py L86-94; archived 2025-11-21)
  - https://github.com/Plachtaa/seed-vc/blob/main/EVAL.md
  - arXiv 2411.09943
- **RVC:**
  - https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI (README L21, L87-102; requirments_cu128_py312.txt; docs/cn/faq.md Q9-Q11; i18n/locale/zh_CN.json L84, L257)
  - RTX 50 issue: https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/issues/2745
  - 50-series package mirror: https://huggingface.co/SaltProphet/VoiceConversionWebUI/blob/main/RVC20240604Nvidia50x0.7z
- **Other VC and enhancement tools:**
  - so-vits-svc: https://github.com/svc-develop-team/so-vits-svc
  - so-vits-svc-fork: https://github.com/voicepaw/so-vits-svc-fork (README "No Longer Maintained")
  - Applio: https://github.com/IAHispano/Applio
  - OpenVoice: https://github.com/myshell-ai/OpenVoice
  - MeanVC: https://github.com/ASLP-lab/MeanVC
  - MeanVC2: https://github.com/ASLP-lab/MeanVC2
  - Vevo: https://github.com/open-mmlab/Amphion (models/vc/vevo); weight licences from HF tags at https://huggingface.co/amphion/Vevo1.5 (search snippet)
  - Chatterbox: https://github.com/resemble-ai/chatterbox
  - resemble-enhance: https://github.com/resemble-ai/resemble-enhance
  - ClearerVoice-Studio: https://github.com/modelscope/ClearerVoice-Studio
  - AP-BWE: https://github.com/yxlu-0102/AP-BWE
- **Papers (search snippets only):** VoxSim arXiv 2407.18505; YourTTS 2112.02418; PASE 2511.13300; UniPASE 2604.14606.
- **Local files:**
  - `/home/user/voicetwin/research/一模一样/记录.md` L104, L106
  - `/home/user/voicetwin/research/gsv_v4_v5/记录.md` L112
  - `/home/user/voicetwin/voicetwin/synth/engine.py` L506, L1061, L2278-2340, L2840
  - `/home/user/voicetwin/voicetwin/eval/speaker.py` L90, L109
  - `/home/user/voicetwin/research/gsv_v4_v5/官方资料/v5整合包/runtime_包版本.txt`
- **Not reachable from this session:** arxiv.org, aclanthology.org, alphaxiv.org, huggingface.co, api.github.com. Paper and Hugging Face figures above therefore come from search-result snippets and could not be checked against the full text. I could not read the KIT IWSLT 2026 TTS→VC cascade paper (2606.07240), which may hold the only direct cascade numbers.
- **Downloaded copies** of the READMEs and code: `/tmp/claude-0/-home-user/3d87cc6c-41dd-58c2-802f-d51dc0c78527/scratchpad/dl_seedvc`, `/tmp/claude-0/-home-user/3d87cc6c-41dd-58c2-802f-d51dc0c78527/scratchpad/dl_gsv`, `/tmp/claude-0/-home-user/3d87cc6c-41dd-58c2-802f-d51dc0c78527/scratchpad/dl_misc`.
