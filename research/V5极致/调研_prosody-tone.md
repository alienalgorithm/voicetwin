# 调研：prosody-tone（2026-11-24）

REPORT: Tone (语气), intonation, rhythm and speaking habits on GPT-SoVITS V5. Research only; no files in /home/user/voicetwin were changed.

Network note: arxiv.org, huggingface.co, yuque, modelscope and openreview cannot be reached from this session. Official code and READMEs were read directly from raw.githubusercontent.com. Papers I could not open are cited by URL, with the claim taken from search-result summaries (marked [search summary]). Everything marked [judgement] or [untested] has no published number behind it.

## 0. Bottom line
1. In V5, the per-sentence reference clip is the strongest inference-time lever for tone.
   - Only one reference is used.
   - The SoVITS/CFM stage sees only its last ~5 s.
   - The official wiki says v3/v4-family output "lean[s] more toward the reference audio rather than the overall training set" (V5 is built on v4).
   - So better per-sentence reference retrieval is the highest-value prosody change. It needs no GPU and no training.
2. The fine-tuned GPT (s1, text to semantic) is the only place her long-term habits are learned from all of her material. Official GPT-SoVITS DPO does not target prosody: its "rejected" samples are random repeats or deletions of semantic tokens. Prosody gains must come from:
   - clean, verbatim training text;
   - choosing the GPT checkpoint by prosody metrics against her held-out recordings.
3. VoiceTwin already has most of the plumbing:
   - twin_profile (pause, F0 and rate statistics per group);
   - the reference bank and match_score;
   - an F0-DTW term in select.py;
   - leave-one-out weight calibration in calibrate_rank.py.
   But the prosody weights default to 0 and the pause/final-contour statistics are not used to rank candidates. Turning these on through the existing calibrate_rank.py is low-risk and CPU-only.
4. One V5 knob is not used today: cfg_rate. V5dev defaults to 1.30. VoiceTwin never sends it. The official V5 package's api_v2.py accepts it.
5. The cheapest guaranteed "identical" output: when a script sentence matches something she already recorded, use her real recording. This is opt-in. Nothing in VoiceTwin does this today.
6. Other open-source TTS systems do not beat V5 on published zero-shot similarity at sizes that fit her GPU (RTX 5070 Laptop, 8 GB, about 6.8 GB free, handbook §六). None publish fine-tuned single-speaker prosody results that compare with V5. At most, test one of them (VoxCPM1.5 or Fun-CosyVoice3-0.5B) offline as a fallback. Do not integrate one now.

## 1. How V5 actually produces tone (verified in code)
Line numbers are from GPT-SoVITS main, fetched 2026-11-24. The V5 package files differ slightly in size from main: models_v5.py is 3748 B in the package vs 3785 B on main, and TTS.py is 90195 B vs 92976 B (from `research/gsv_v4_v5/官方资料/v5整合包/文件清单.tsv`). Re-check these constants in the package (e.g. via gsv-v5-package.yml).

- **Reference length and GPT prompt.**
  - `TTS_infer_pack/TTS.py` L907–933: the reference must be 3–10 s.
  - 0.3 s of zeros is appended before HuBERT. The whole reference becomes `prompt_semantic`, the prefix that the GPT continues.
  - So the GPT continues prosody from the end of the reference.
- **V5 SoVITS uses one reference, last 5 s only.**
  - TTS.py L1783–1816 (`using_vocoder_synthesis`): only `refer_spec[0]` is used.
  - `module/models_v5.py`: `V5_REFERENCE_FRAMES=500` (L10). The mel hop is 320 at 32 kHz (TTS.py L159–171), i.e. 100 frames/s, so only the last 5.0 s of the reference conditions the CFM.
  - Targets are generated in chunks of up to 640 frames with a 32-frame rolling tail (L51–84). Noise temperature is fixed at 0.875 (L7).
- **What cfg_rate does.**
  - Default is 1.30 for v5dev and 0 for v5turbo (TTS.py L1788).
  - The CFG negative branch drops the audio prompt but keeps the text (`module/models.py` L1171–1191; `cfg_drop_text=False` for V5, models_v5.py L20).
  - So a higher cfg_rate pushes output toward what the reference implies [code reading; effect size untested].
  - The package's api_v2 accepts it: local `research/gsv_v4_v5/官方资料/v5整合包/api_v2.py` L181, L484, L513.
  - VoiceTwin's payload (`voicetwin/backends/gptsovits.py` L3915–3935) does not send it.
- **GPT training sees one utterance at a time.**
  - `AR/models/t2s_model.py` L355 and L408: input is phones + BERT features, target is the same utterance's semantic tokens. No cross-utterance context.
  - [judgement] Habits that span a sentence boundary (prompt sentence → target sentence) are learned only if training clips contain more than one sentence.
- **Official DPO.**
  - t2s_model.py L414–444; `AR/models/utils.py` L244–282 (`make_reject_y`): the "rejected" sample is a random span repeat (`repeat_P`) or deletion (`lost_P`).
  - This targets repetition and skipped words, not prosody. It also doubles VRAM (handbook §10.3). Keep it off.
- **Official wiki** (GPT‑SoVITS‑features page, V2Pro section): "the synthesized tone and timbre of v3/v4 lean more toward the reference audio rather than the overall training set". The same page credits v3 with richer emotion and more stable GPT output.

## 2. What VoiceTwin already does (so we don't duplicate it)
- `synth/search.py` `match_score` L150–162 scores reference candidates on:
  - kind (statement, question or exclaim; `utils/textutil.py` L120);
  - syllable-length ratio;
  - English ratio;
  - paragraph-initial;
  - self-similarity prior;
  - long-pause penalty.
  Refs with the same text as the target are excluded (L182).
- `eval/identical_judge.py`: candidate scoring already uses duration z-score against her rate model and her p97 mid-sentence pause. F0-shape and LTAS terms exist (`_shape` L326–352) but `rank_w` is pros=0, ltas=0 (L93).
- `synth/calibrate_rank.py`: leave-one-out, bootstrap-guarded adoption of ranking weights against held-out real recordings. Grid: pros 0/0.15/0.3, ltas 0/0.04 (L23–30).
- `synth/select.py`: checkpoint selection includes DTW-aligned F0 correlation vs her real recording (`_f0_dtw` L461–479) with weight 0.3, plus |ln duration ratio| with weight 0.8 (`item_S`, about L994–1020).
- `style/twin_profile.py`: already measures per-punctuation pause quantiles, P(pause|punct), hesitation rate, F0 groups (statement/question × English bins, including the last-300 ms rise/fall) and an NNLS rate model.

## 3. Recommendations, ranked

### R1. Better per-sentence reference retrieval (V5's main tone lever)
**What to change** (search.py `match_score` / `shortlist_refs`; twin_profile for statistics):
- (a) Finer sentence-function classes, derived from text with regex:
  - tag/rhetorical question (…对不对/是不是/对吧？);
  - enumeration (首先/第一/其次/最后);
  - definition or explanation (就是说/也就是/叫做);
  - emphasis or imperative (注意/一定/千万, ！);
  - reading an English example (a span of N or more English words).
  Keep only classes whose F0, rate or final-contour statistics in HER data really differ from plain statements. twin_profile can test this, which keeps the classes data-driven.
- (b) Text similarity between reference and target: shared grammar terms from the built-in term list, plus character-bigram overlap. This is a cheap stand-in for context-aware retrieval.
- (c) A V5-specific end check. Because only the last 5 s reaches the CFM and the GPT continues from the reference's end, prefer refs whose final 5 s are continuous speech ending in a complete sentence of the right type. The bank already requires a complete sentence.
- (d) Test whether "length close to target" is actually right. Short targets currently pull 3 s refs, which give the GPT little context.
  - Longer prompts carry more prosodic habit [search summary]: LP-TTS-Bench, arXiv 2610.19774, reports prosody gains growing with prompt length up to ~10 min; Mega-TTS 2, arXiv 2307.07218, says one sentence is too little.
  - Add "prefer 5–10 s refs" as a calibration option, not a hard rule.
- (e) [untested] Also offer contiguous two-sentence spans from her source recordings (≤9.8 s) as bank entries. These are natural rather than spliced.

**Evidence:**
- The wiki quote and the code facts in section 1.
- Xue et al., Interspeech 2024, arXiv 2406.03714: context-aware retrieval (CA-CLAP) beats random selection and text-only retrieval [search summary; numbers not read].
- No GPT-SoVITS-specific numbers exist.

**Cost:** CPU only. Current shortlist takes 3.2–3.9 ms per sentence (`research/一模一样/记录.md` P5). Generation time is unchanged because arms stay within the existing 40–64 candidate cap.

**Risk:** proxy overfitting. Gate adoption through calibrate_rank (see R2). No effect on silence.

### R2. Prosody likelihood under HER distribution in candidate ranking (identical_judge + calibrate_rank)
**What to change:**
- Add z-scores against her own statistics, so the target is "inside her natural range", not "match the mean":
  - pause at each punctuation mark: presence vs P(pause|punct), and duration vs her quantiles;
  - pauses not at punctuation vs her hesitation rate per 100 syllables;
  - final 300 ms F0 rise/fall vs her group;
  - F0 range and declination vs her group.
- Add these to `calibrate_rank.grid()`.
- Extend the calibration ground truth to include pause alignment vs the real recording, alongside today's speaker term, F0-contour term and length term.
- Turn on `pros` only if the existing rule adopts it: at least +0.05 leave-one-out Spearman and 99% of bootstrap resamples better.

**Evidence:** the data structures already exist. The adoption rule is already proven against chance (the 26.5% random-pass finding is recorded in calibrate_rank's docstring). Benefit size is [untested] and is measured by this same mechanism on her PC.

**Cost:** the shape terms take about 45 ms per fully scored candidate (P5 note 11), so about 1 s per sentence at 24 full scores. `align_pauses` is milliseconds.

**Risk:** low, because adoption is automatic only when proven. Signal path untouched.

### R3. Reference-aware prosody metrics for checkpoint selection, especially the GPT checkpoint (select.py)
**What to change:**
- Add pause alignment vs her real recording of the same held-out sentence: position match plus duration ratio. This follows Seamless' local_prosody toolkit, whose README reports `mean_alignment_score` and `duration_score`.
- Add an energy-contour correlation on the existing DTW path.
- [untested] Add a semantic-token edit distance between the generated audio and her real clip. Re-encode the generated audio with the same HuBERT + quantizer used in step 1C. Her real tokens already exist from training extraction.
  - Precedent [search summary]: DS-WED, arXiv 2509.19928, a weighted edit distance on SSL k-means tokens, correlates 0.77 with human prosody MOS vs 0.30 for log-F0 RMSE and 0.66 for MCD (paper table).
  - This isolates the GPT's rhythm and phrasing from SoVITS timbre.
- Rank GPT checkpoints mainly on these prosody metrics and SoVITS checkpoints mainly on timbre.
- Classic F0 metrics (GPE, VDE, FFE) are from Skerry-Ryan et al. 2018, arXiv 1803.09047.

**Cost:** CPU. HuBERT re-encode time on CPU not measured.

**Risk:** about 20 held-out sentences is small. Keep the bootstrap/"明显" logic already in lang_groups.

### R4. Make cfg_rate (and possibly sample_steps) a search dimension
**What to change:**
- Add `cfg_rate` to the payload, default None = official 1.30.
- Let `plan_arms` include e.g. {1.0, 1.3, 1.6}, or let the R2 calibration pick one value per voice.
- Keep v5dev (32 steps). On few-step flow-matching models, guidance at the first (pure-noise) step erodes speaker similarity (arXiv 2611.11179 [search summary]: CosyVoice 2 −0.5 SIM at 4 steps, −5.8 at 2 steps). That matters for v5turbo only if cfg > 0 is forced.

**Cost:** none extra inside the cap.

**Risk:** [judgement] a high CFG can over-saturate or add artifacts. Let the scorer reject those. First verify on GitHub's real-package CPU run that the API accepts the field.

### R5. Training text = exactly what she said (export-time only)
**What to change:**
- (a) Fillers or repetitions present in the audio but missing from the transcript. Whisper-style ASR drops fillers: CrisperWhisper, arXiv 2408.16589, documents this for Whisper, but that model is English/German and CC-BY-NC, so it is only a reference point. Missing fillers teach the GPT to insert them unpredictably [judgement].
  - Flag clips where the ASR-aligned speech time is much longer than the text explains. Paraformer-zh returns timestamps per FunASR README L184.
- (b) [untested, needs teacher consent] In the exported training text only, insert "，" where her measured pause is at least her comma p25 and no punctuation exists. The GPT then learns "comma → her pause", and her scripts' punctuation reproduces her phrasing.
  - Her proofread table and the 母本 stay untouched. Because of the 母本-first rules, present this to her as a decision.

**Cost:** CPU, at data preparation.

**Risk:** a wrong comma insertion teaches false pauses. Use high thresholds and log every change.

### R6. Real-recording reuse (opt-in "用你的原声")
**What to change:** when a normalized script sentence equals a confirmed training clip (complete sentence, not cut mid-sentence), use her real clip. Trim it and join it with the same digital-silence logic. Report "原声 n 句" in the result report. Today, same-text clips are deliberately excluded as references; this would be a separate feature.

**Evidence:** identical by definition. How often it applies is unknown. Measure it as "% of script sentences with an exact match", CPU only. Her lectures follow the 母本, so the rate could be meaningful [judgement].

**Risk:**
- Mixing real and synthetic sentences may make the synthetic ones stand out.
- Within-sentence pauses keep her studio floor. This needs a decision against the absolute-silence rule, which is written about generated audio.
- Never splice phrases inside a sentence (seams).

### R7. Her habitual phrasing and fillers in scripts: suggestions only
**What to change:** mine her transcripts for sentence-initial discourse markers (那么/好/我们来看/也就是说/对吧). Offer "her usual way" suggestions in the script editor, and never apply them automatically, per her text rules.

**Evidence:** Matsunaga et al., APSIPA 2022, arXiv 2210.07559: personalized filled pauses need correct placement for naturalness and correct word choice for individuality [search summary]. Group-wise personalised prediction beats non-personalised (LREC 2022) [search summary].

**Risk:** she may not want fillers in lectures. Ask her.

### Not recommended now
- **Official DPO:** only repeat/omission corruption; doubles VRAM on an 8 GB laptop.
- **Custom preference or RL training on the GPT.** The closest fit is SpeechAlign (NeurIPS 2024, arXiv 2404.05600): real ("golden") tokens preferred over the model's own tokens. GRPO with similarity/CER/emotion rewards is also possible; GLM-TTS ships `grpo/` code, Apache-2.0, for a similar LLM + flow-matching architecture.
  - Preference alignment improves WER and SIM in large LM-TTS (Tian et al., arXiv 2409.12403 [search summary]).
  - But nothing is published on GPT-SoVITS or on single-speaker prosody, it requires forking s1 training, and VRAM is tight. A later research option, not a v18.7 item.
- **emotion2vec similarity as a ranking signal.** It is common (IndexTTS2 uses it as "ES"). But arXiv 2604.26347 ("The False Resonance") finds it rewards acoustic mimicry and does not track emotion [search summary]. Log it at most.
- **AutoPCP** (Seamless, arXiv 2312.05187; code in facebookresearch/stopes `eval/auto_pcp`, which scores prosodic similarity from 1 to 4). Weight licence not verified. Possibly useful only as an offline check in R3.

## 4. Objective prosody metrics we can compute automatically
| Metric | Needs her real clip of the same text? | Use | Source |
|---|---|---|---|
| DTW log-F0 corr/RMSE; GPE/VDE/FFE | yes | select.py (corr/RMSE exist) | Skerry-Ryan 2018 arXiv 1803.09047 |
| Pause alignment (position + duration), speech-rate ratio | yes | select.py, calibrate_rank ground truth | stopes `eval/local_prosody/README.md` (Seamless) |
| Semantic-token edit distance (DS-WED-style) | yes | GPT checkpoint choice | arXiv 2509.19928 (r = 0.77 with prosody MOS [search summary]) |
| Pause@punct likelihood, hesitation rate, final-contour z, F0 range/declination z vs her groups | no | identical_judge ranking | twin_profile (existing) |
| Duration z vs her rate model | no | exists | identical_judge |
| TTScore-pro (FACodec prosody-token likelihood) | no | optional research | arXiv 2509.20485, github.com/rsmlgen/TTScore (licence unchecked) |
| UTMOSv2 naturalness | no | sanity check only, not speaker-specific | github.com/sarulab-speech/UTMOSv2 (MIT) |

## 5. Other open-source TTS as possible assistants
Seed-TTS-eval test-zh: CER% / SIM%, using the WavLM-large SV model per the seed-tts-eval README. Sources: CosyVoice README "Evaluation" table, VoxCPM README "Seed-TTS-eval" table, and the official GPT-SoVITS chart (`research/gsv_v4_v5/官方资料/零样本相似度对比图.png`). Cross-table values for the same model can differ by about 2 SIM points (F5-TTS: 74.1 in the CosyVoice table vs 76.0 in VoxCPM's).

| Model | test-zh CER / SIM | Size | Licence | Fine-tune on ~2.5 h | Fits ~6.8 GB free? |
|---|---|---|---|---|---|
| GPT-SoVITS v5dev (official chart) | – / 80.5 (zero-shot) | 0.38B | MIT (GSV) | yes (what we do) | yes |
| LongCat-Audio-DiT | 1.09 / 81.8 | 3.5B | not checked | not checked | doubtful |
| VoxCPM2 | 0.97 / 79.5 | 2B | Apache-2.0 | SFT + LoRA | ~8 GB VRAM per README → likely no |
| VoxCPM1.5 | 1.18 / 77.0 | 0.8B | Apache-2.0 | SFT + LoRA | ~6 GB → yes |
| Fun-CosyVoice3-0.5B-2512 (RL) | 1.21 / 78.0 (0.81 / 77.4) | 0.5B | Apache-2.0 | recipes in examples/; GRPO for CV2 | VRAM not stated |
| MOSS-TTS | 1.20 / 78.8 | n/a | not checked | n/a | n/a |
| Qwen3-TTS-1.7B (0.6B also Base) | 1.22 / 77.0 | 1.7B | Apache-2.0 | single-speaker SFT script | not stated |
| IndexTTS2 (2.5 released 2026-08-10) | 1.03 / 76.5 | 1.5B | bilibili licence; §3.4(c) bars using it to improve other non-bilibili AI models unless those models are non-commercial | no official script | not stated |
| GLM-TTS (RL) | 1.03 / 76.1 (0.89 / 76.4) | 1.5B | Apache-2.0 | SFT / LoRA + GRPO code | not stated |
| F5-TTS | 1.52–1.53 / 74.1–76.0 | 0.3B | weights CC-BY-NC | finetune gradio | yes |
| Spark-TTS | 1.2 / 66.0 | 0.5B | Apache | training code not released | yes |
| MegaTTS3 | 1.52 / 79.0 | 0.5B | Apache | WaveVAE encoder withheld, so cannot clone her voice locally | n/a |
| Fish S2-Pro / Drama 3 | 0.54 CER / – | 4B / 35B-A3B | Fish community/research licences | n/a | no |

**Verdict:**
- Zero-shot, V5 is already at or above every open model here that fits her GPU.
- Prosody claims for the others (emotion control, duration control, instruct tags) are about controllability, not imitating one speaker's habits.
- Her material has almost no full-English sentences, so "use X for English" has little value.
- Mixing engines inside one lecture risks timbre and bandwidth jumps. VoiceTwin already rejected V4 48 kHz / v2ProPlus 32 kHz mixing for exactly this reason (`research/一模一样/记录.md`, 淘汰的做法).
- At most: a one-off offline bake-off on her PC with VoxCPM1.5-LoRA or CosyVoice3-0.5B, using the same held-out set and the same scorer. Integrate only if it measurably beats V5 on her voice.

## 6. How to verify
- **GitHub Actions (CPU):**
  - unit tests for the new features, using synthetic clips with known pauses and F0, like `tests/test_twin_profile.py`;
  - in `gsv-v5-real.yml`, confirm the real V5 package accepts `cfg_rate` and the pipeline still runs;
  - count script sentences with exact matches for R6.
  Actions cannot judge her prosody.
- **Teacher's PC:**
  - run generation on the held-out set with the old vs new reference selection and ranking;
  - compute the section 4 reference-aware metrics against her real recordings;
  - adopt only through calibrate_rank's existing rule (+0.05 leave-one-out and 99% bootstrap);
  - then a short blind A/B by her ear. HONEST_NOTE already says the ear decides.
  - Report the measured numbers only.

## 7. Open uncertainties
- The constants above are from GPT-SoVITS main. Check `V5_REFERENCE_FRAMES`, the cfg default and the noise temperature in the package's own files.
- No published effect sizes exist for any of R1–R7 on GPT-SoVITS.
- Licences for AutoPCP weights, DS-WED, emotion2vec and TTScore were not verified.
- The Seed-TTS speaker fine-tuning detail could not be read (arXiv blocked). The search summary only says fine-tuning gave "even higher subjective scores".

## Sources
- GPT-SoVITS code (main): https://raw.githubusercontent.com/RVC-Boss/GPT-SoVITS/main/GPT_SoVITS/TTS_infer_pack/TTS.py , .../GPT_SoVITS/module/models_v5.py , .../GPT_SoVITS/module/models.py , .../GPT_SoVITS/AR/models/t2s_model.py , .../GPT_SoVITS/AR/models/utils.py
- GPT-SoVITS wiki features: https://github.com/RVC-Boss/GPT-SoVITS/wiki/GPT%E2%80%90SoVITS%E2%80%90features-(%E5%90%84%E7%89%88%E6%9C%AC%E7%89%B9%E6%80%A7)
- GPT-SoVITS changelog: https://raw.githubusercontent.com/RVC-Boss/GPT-SoVITS/main/docs/cn/Changelog_CN.md
- Local files: /home/user/voicetwin/research/gsv_v4_v5/官方资料/v5整合包/api_v2.py , /home/user/voicetwin/research/gsv_v4_v5/官方资料/零样本相似度对比图.png , /home/user/voicetwin/research/gsv_v4_v5/记录.md , /home/user/voicetwin/research/一模一样/记录.md , /home/user/voicetwin/教学手册.md
- seed-tts-eval: https://github.com/BytedanceSpeech/seed-tts-eval
- CosyVoice: https://github.com/FunAudioLLM/CosyVoice
- VoxCPM: https://github.com/OpenBMB/VoxCPM
- IndexTTS (README and LICENSE): https://github.com/index-tts/index-tts
- Qwen3-TTS (README and finetuning/README.md): https://github.com/QwenLM/Qwen3-TTS
- GLM-TTS: https://github.com/zai-org/GLM-TTS
- Fish Speech: https://github.com/fishaudio/fish-speech
- F5-TTS: https://github.com/SWivid/F5-TTS
- Spark-TTS: https://github.com/SparkAudio/Spark-TTS
- MegaTTS3: https://github.com/bytedance/MegaTTS3
- MOSS-TTSD: https://github.com/OpenMOSS/MOSS-TTSD
- FunASR (Paraformer timestamps, README L184): https://github.com/modelscope/FunASR
- stopes (local_prosody, auto_pcp): https://github.com/facebookresearch/stopes
- UTMOSv2: https://github.com/sarulab-speech/UTMOSv2
- Papers (search summaries only):
  - context-aware prompt retrieval: https://arxiv.org/abs/2406.03714
  - long-prompt prosody benchmark: https://arxiv.org/abs/2610.19774
  - Mega-TTS 2: https://arxiv.org/abs/2307.07218
  - DS-WED: https://arxiv.org/abs/2509.19928
  - TTScore: https://arxiv.org/abs/2509.20485
  - SpeechAlign: https://arxiv.org/abs/2404.05600
  - preference alignment for LM-TTS: https://arxiv.org/abs/2409.12403
  - few-step CFG and speaker similarity: https://arxiv.org/abs/2611.11179
  - emotion2vec critique: https://arxiv.org/abs/2604.26347
  - personalised filled pauses: https://arxiv.org/abs/2210.07559
  - CrisperWhisper: https://arxiv.org/abs/2408.16589
  - Seamless: https://arxiv.org/abs/2312.05187
  - Seed-TTS: https://arxiv.org/abs/2406.02430
  - CosyVoice 3: https://arxiv.org/abs/2505.17589
  - IndexTTS2: https://arxiv.org/abs/2506.21619
  - Skerry-Ryan 2018: https://arxiv.org/abs/1803.09047
