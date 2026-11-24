# 调研：voicetwin-gaps（2026-11-24）

VoiceTwin V5 fidelity: gap analysis (research only; no files in /home/user/voicetwin were edited)

Sources. I read the VoiceTwin code and docs listed in the task. I also downloaded official GPT-SoVITS **main-branch** files from raw.githubusercontent.com on 2026-11-24 and saved them under /tmp/claude-0/-home-user/3d87cc6c-41dd-58c2-802f-d51dc0c78527/scratchpad/gsv_main/: TTS.py, models_v5.py, models.py, s2_train_v3_lora.py, data_utils.py, AR/data/dataset.py, lr_schedulers.py, s1longer-v2.yaml, s2.json, 2-get-hubert-wav32k.py and api_v2.py.

**Caveat on versions.** I compared byte sizes against the teacher's package file list (research/gsv_v4_v5/官方资料/v5整合包/文件清单.tsv).
- These files are the same size in main and in the package, so they are very likely identical: s2_train_v3_lora.py (13000), data_utils.py (41206), dataset.py (12547), lr_schedulers.py (2618), s1longer-v2.yaml (517), s2.json (1581).
- These files differ in size, so the V5 constants I quote from main must be re-checked against the package copies: models_v5.py (package 3748 vs main 3785), TTS.py (90195 vs 92976), models.py (57586 vs 58260).
- Main's api_v2 has also gained cache_dit and gpt_streaming fields. The package api_v2 does **not** have them, but it **does** accept `cfg_rate` (research/gsv_v4_v5/官方资料/v5整合包/api_v2.py:180-181).

## A. Levers VoiceTwin already uses for V5 (file:line)

### Model and training
1. **Version.** v5dev is the default and the only version trained; `compare_versions` is none (voicetwin/backends/gptsovits.py:106-111; research/gsv_v4_v5/记录.md:93).
2. **SoVITS training uses the official LoRA path.**
   - Script: s2_train_v3_lora.py (gptsovits.py:2808-2810), with the s2.json template (2799-2800).
   - No discriminator (2824), `grad_ckpt` False (2827).
   - `lora_rank` is 32 by default; 16/32/64/128 are allowed (2803-2806, 1043-1044; default_config.yaml:97).
3. **LoRA batch, epochs and saving.**
   - Batch follows the official formula (VRAM+0.4)//8, capped at n_clips/4 (gptsovits.py:1030-1037). On her RTX 5070 Laptop 8 GB card (教学手册.md:188) this gives 1.
   - Epochs: 8 in identical mode, 4 in standard, maximum 16, a checkpoint saved every epoch (1009-1013, 1038-1042).
   - Selection tests every LoRA epoch from 2 onward (97, 3531).
4. **GPT (s1).** Full fine-tune from s1v3.ckpt with the official s1longer-v2.yaml (87-88, 2846-2860).
   - Identical mode: 20 epochs at batch ≤4, scaled up for bigger batches to keep the same number of updates, maximum 50 (259-262, 1090-1095).
   - GPT batch is measured on the GPU (PROBE_GPT = (4,6,8,12), line 271).
   - DPO is off.
   - In the official code the GPT learning rate is effectively a constant 0.002: lr_schedulers.py `step()` hard-codes `lr = 0.002` ("锁定用线性…直接锁定").
5. **Semantic tokens** are extracted per version with that version's own base model (gptsovits.py:643-644; 记录.md "语义token").
6. **Mixed zh-en text frontend.** English words inside Chinese lines are trained, mirroring inference-time LangSegmenter (voicetwin/backends/gsv_mixed_text.py:1-12; gsv_scripts/get_text_mixed.py:141-175; gptsovits.py:2672-2687).
   - Without it, the official 1-get-text deletes Latin letters. VoiceTwin's own count: 559 of 1004 lines, 2080 English words (gsv_mixed_text.py:4).
   - Pure-English lines (lang=en) go through the official English g2p with no BERT (get_text_mixed.py:165-173).
7. **Trained-share check.** Logs are parsed for official loader drops: s1 phones/s outside [3,25] or longer than 54 s; s2 clips ≤0.6 s or ≥54 s. Training warns or stops if fewer than 98% of clips were trained (gptsovits.py:282, 1220-1275, 1508-1540, 1567).

### Selection
8. `select_deep` runs in three steps (voicetwin/synth/select.py:1345-1460):
   - **Step A** screens GPT checkpoints with SoVITS fixed at the epoch nearest half of the maximum (1378), weights w_t=1, w_c=4 (1382).
   - **Step B** screens SoVITS checkpoints, w_t=1.5, w_c=3 (1392).
   - **Step C** is a 3×3 final with 2 references × 4 candidates (constants 362-368).
   - Per-item score (993-1019) = speaker similarity (3 SV models) + to_real vs. the real held-out clip − 3·CER − 0.8·|ln duration ratio| − 0.3·(1 − r_F0).
   - Validation set is 20 clips (default_config.yaml:45; prepare.py:578-604), plus 24 test texts.
   - Speed is recalibrated, clipped to 0.8–1.25 with a 3% deadband (select.py:372-374).
   - Reference audition: 24 refs × 6 items × 4 (376-379).

### Inference payload sent to api_v2 for V5 (gptsovits.py:3897-3936)
9. Fields and values:
   - `text_lang`: zh whenever the text has any CJK character (utils/textutil.py:38-44).
   - `top_k` / `top_p` / `temperature`: per-arm presets {1.0, 15, 1.0} and {1.0, 5, 1.0}; rescue preset {0.6, 15, 0.8} (synth/engine.py:154-156). Config defaults are 15 / 1.0 / 1.0 (default_config.yaml:100-102).
   - `repetition_penalty`: 1.35 (3928).
   - `text_split_method`: cut0 (3920).
   - `batch_size`: n duplicated lines; V5 rows are synthesized independently (341-345).
   - `fragment_interval`: 0.3, or 0.5 when batching.
   - `seed`: per candidate.
   - `speed_factor`: calibrated value (3907).
   - `parallel_infer`: True (3927).
   - `super_sampling`: False (3929). This is a no-op for V5: TTS.py applies it only when version == v3 (TTS.py:1589).
   - `sample_steps`: not sent in auto mode (3933-3935), so the engine uses 32 for v5dev (TTS.py:1203).
   - `cfg_rate`: **never sent**, so v5dev uses 1.30 (TTS.py:1204; models_v5.py:54).
   - `aux_ref_audio_paths`: empty for V5 because `supports_aux_refs` is False for v3/v4/v5 (gptsovits.py:4004-4011; engine.py:863, 1030).

### Reference choice and candidates
10. **Per-sentence references (identical mode).**
    - 3 refs per sentence (engine.py:152), taken from the reference bank: clips 3.2–9.8 s after trimming, with 80 ms of edge kept (data/references.py:195-196, 360).
    - Ranked by `match_score` (synth/search.py:150-162): sentence-kind match, |ln syllable ratio|, English-share match, paragraph-initial, audition prior, long-pause penalty.
    - The second ref is forced to come from a different video (search.py:189-193). An English-bearing ref is added if the sentence is ≥15% English (199-202).
    - The audition prior is a z-score shrunk 50% toward the base score (search.py:129-141). The base score prefers about 6.5 s length (references.py:33-35).
11. **Candidates per sentence.** 40–64 per sentence on high/mid GPUs (engine.py:149-163). Arms are refs × 2 presets; the aux dimension collapses for V5 (search.py:260-277). Scoring is quick then full; the whole lecture is re-ranked for continuity.

### Prosody, loudness and sample rate
12. **Pauses, pitch and loudness habits** come from style/twin_profile.py: pause distributions per punctuation measured on the raw recordings, pitch slope and final rise or fall, a speech-rate model, loudness and onset/offset. Sentences are joined with digital zeros.
13. **Loudness.**
    - Normalized to -20 LUFS per whole source file before slicing (data/prepare.py:695; data/enhance.py:224), so loudness variation inside a lecture is preserved.
    - The official 1B step adds a per-clip 50% peak mix (2-get-hubert-wav32k.py:60-88).
    - Final output is matched to the source LUFS (engine.py:2848-2865).
14. **Sample-rate path.** Clips at 44.1 kHz (default_config.yaml:14) → official 32 kHz / 16 kHz features → V5 vocoder output at 48 kHz (TTS.py:755). Concatenation keeps the highest rate (engine.py:1883, 1906).
    - The vocoder is the pretrained gsv-v5-pretrained/vocoder.pth (gptsovits.py:93-94) and is never fine-tuned.
    - Training mel is 32 kHz with fmax None, so the band above 16 kHz in the output is generated by the vocoder, not learned from her recordings. This is inferred from the code; there is no measurement of it.

## B. Gaps: what is not used yet, or uses old defaults (lever | now | change | why)

1. **cfg_rate (V5 CFG strength): the biggest unused lever.**
   - Now: not sent, so it is 1.30 (gptsovits.py:3897-3936).
   - Change:
     - Add `infer.cfg_rate: auto` and pass it through.
     - In identical mode, replace the dead aux arm dimension for V5 (engine.py:153 `aux_options`) with cfg variants, e.g. 1.3 and 2.0.
     - In select_deep step C, sweep {1.0, 1.3, 1.6, 2.0} on the held-out clips.
   - Why: in V5 the negative branch drops only the audio prompt and keeps the text (models_v5.py: `cfg_drop_text=False`; models.py `CFM.inference`: `drop_audio_cond=True`, `v = v + (v − neg)·cfg`). So cfg_rate directly scales how strongly the output is pulled toward the reference voice.
   - Effect size: untested; there are no published V5 cfg sweeps.
   - Cost: with cfg ≤ 1e-5 the second DiT pass is skipped, so cost is about half. Any cfg > 0 costs the same as now.
   - Risk: over-strong CFG may give harsh or metallic artifacts (my judgement). After LoRA fine-tuning the unconditional branch also shifts toward her voice, so the effect may shrink (untested).
   - Verify on her PC with to_real and CER. On Actions, only the request plumbing can be checked (tests/gsv_real/api_v2.py already has `cfg_rate`).

2. **Reference length for V5.**
   - Fact: the V5 CFM uses only the last `V5_REFERENCE_FRAMES = 500` mel frames of the reference (models_v5.py). That is 5.0 s, because mel_fn_v4 runs at 32 kHz with hop 320, i.e. 100 frames/s (TTS.py:159-171).
   - Fact: target chunk length = min(1000 − ref_frames, 640) frames, so 5.0–6.4 s of speech per CFM chunk, with a 32-frame rolling tail.
   - Now: `match_score` penalizes |ln syllable ratio| (search.py:157). Short sentences therefore get short refs (down to 3.2 s), which gives the CFM less timbre context.
   - Change: for V5 only, prefer refs ≥ 5 s whose last 5 s are continuous speech (there is already a max_pause penalty, search.py:161), and relax the syllable-match term.
   - Effect: untested. Verify with an A/B in the audition step.
   - Note: the GPT prompt still uses the full reference and its full text, so the text must match exactly. It does: the bank requires unchanged proofread text.

3. **Reference audition coverage.**
   - Now: only 24 refs are auditioned (select.py:376); other refs keep their base-score prior.
   - Change: for V5, where one ref is the whole speaker condition, audition all bank refs ≥ 5 s (or the top 64) and drop poor ones.
   - Cost: linear; 576 generations now, 2400 for 100 refs (my arithmetic).
   - Effect: untested.

4. **LoRA rank and capacity.**
   - Now: rank fixed at 32 in auto mode. The LoRA targets only the DiT attention (`to_q`/`to_k`/`to_v`/`to_out.0`, alpha = r) (s2_train_v3_lora.py:154-158). bridge, wns1 and ref_enc are trained fully; enc_p, ssl_proj and quantizer are frozen (models.py `freeze_quantizer`; s2.json).
   - Change: in identical mode, also train rank 64 and/or 128 as extra candidates and let select_deep choose. Extending targets to the FFN layers would need patching the official script; I do not recommend that now.
   - Why: she has about 2.4 h of studio material, while the official defaults (2 epochs, "训太多=作死", webui.py:129-131) target small datasets. The wiki says V5 few-shot fine-tuning "暂未进行研究和实验" (记录.md:15).
   - My parameter estimate: rank 32 is about 5.8M trainable DiT params; rank 128 is about 23M. VRAM impact should be small, but training time doubles per extra run. Untested.

5. **Adaptive epochs.**
   - Now: LoRA uses a fixed 8 epochs; GPT uses a fixed 20 (scaled).
   - Change: if selection picks the last saved epoch, continue training (LoRA up to 16, GPT up to 50) and reselect.
   - My estimate at batch 1 with about 1200 clips: 8 LoRA epochs ≈ 9,600 steps.
   - Evidence that more training helps: none. This is a safe rule because selection is already on held-out clips.

6. **GPT selection weights for tone and rhythm.**
   - Now: step A is CER-heavy (w_c=4, select.py:1382). Intonation enters only as 0.3·(1 − r_F0) and 0.8·|ln duration ratio| (1011-1014). `prosody_z` has weight 0 until calibrated (calibrate_rank.py:30).
   - Change: raise the F0/duration terms in step A for the held-out items, or add a semantic-token or F0-contour distance to her real clip. Keep CER as a gate, not a dominant term.
   - Why: the GPT is the part that learns rhythm and intonation; CER-dominated screening may favor flatter epochs. This is my judgement and untested. Also, screening uses only 2 seeds per item (select.py:366).

7. **Material filters (prepare.py:474-576).**
   - Drop reasons: duration below 2.0 s or above 13 s; whisper avg_logprob < -1.0; no_speech > 0.6; clipping > 0.2%; speech rate outside 0.5–1.8× the median; speaker-similarity outlier.
   - Gap: ASR-confidence drops are skipped only when she edited the text (prepare.py:499-500). Clips she **confirmed without editing** are still dropped.
   - Change:
     - Treat rows covered by "✅ 确认训练素材" as human-verified for the logprob and no_speech filters.
     - Consider min_duration of about 1.0 s for training clips only (the official V4 loader accepts > 0.6 s), to keep short habitual phrases.
   - Share dropped on her data: unknown. Read `prepare_summary.json["dropped"]` on her PC (prepare.py:836-842, 870).
   - If max_duration is ever raised: the official bucket sampler silently drops clips longer than 1000 frames (20 s at hop 640) or ≤ 32 frames (0.64 s) (s2_train_v3_lora.py:107-128; data_utils.py:996-1068). VoiceTwin's trained-count check does not see sampler drops.

8. **English safety.**
   - Now: if the mixed frontend fails, training continues with the official frontend, which deletes English from about 55% of lines (gptsovits.py:2676-2687). Only a warning is shown.
   - Change: stop with a clear message instead, as the missing-module case already does (2694-2703).
   - Why: she requires English to be identical too.
   - Data gap: there are almost no full English sentences (prepare.py:852-861 warns about this). English BERT features are zero in both training and inference, which is official behaviour. Only more English recordings can fix the full-English-sentence gap.

9. **Session consistency.** Old 2 h plus new 26 min recordings may differ in mic or room (unverified).
   - Now: the shortlist deliberately picks a second ref from another video (search.py:189-193).
   - With V5's in-context prompt, the ref sets the acoustic condition per sentence.
   - Change: for V5, prefer refs from one session per lecture, or add session to the continuity term. My judgement; untested.

10. **Long sentences.**
    - Now: max_units_zh is 50 (default_config.yaml:155). Longer sentences are split into separate requests, so the GPT context resets.
    - V5 already chunks long targets internally with a rolling prompt (models_v5.py).
    - Change: measure split vs. unsplit for 50–80-syllable sentences on V5 before changing the limit. Untested.

11. **Wideband timbre is not measured.**
    - Now: the SV models work at 16 kHz; `ltas_d` has weight 0 (metrics.py:468).
    - Change: add a measurement-only comparison of the 4–16 kHz LTAS against her recordings, as a tie-break or report item in LoRA selection. No EQ: that was rejected earlier (research/一模一样/记录.md:104).

12. **sample_steps.** Now auto, so 32. A step-embedding-free DiT allows any count (models.py:1316-1317), so step C could try 48 or 64. Cost is linear. Expected value is small (judgement, untested).

13. **Minor / no change needed.**
    - `repetition_penalty` 1.35, `top_k` 15, `top_p` 1 and `temperature` 1 are the official defaults.
    - `super_sampling` is irrelevant for V5.
    - `parallel_infer` True is fine.
    - The validation set (20 clips ≈ 1.7% of about 1200) was deliberately not enlarged (一模一样/记录.md:107).
    - Vocoder fine-tuning toward her 44.1 kHz recordings is not in the official toolkit. It would be custom work and high risk.

**Verification plan.** Everything that affects fidelity needs real GPU runs on her RTX 5070 Laptop (8 GB). Run every new candidate (cfg_rate, ref policy, rank) through the existing select_deep held-out scoring. GitHub Actions (CPU only) can only check that payloads and configs reach the real package api_v2 and scripts (the existing gsv-v5-real.yml).

**First step before acting on items 1–2.** Extract GPT_SoVITS/module/models_v5.py and TTS_infer_pack/TTS.py from the package in the gsv-v5-package workflow and confirm `V5_REFERENCE_FRAMES`, the cfg default and the chunking constants.
