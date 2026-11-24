# 调研：v5-official（2026-11-24）

# GPT-SoVITS V5: everything the official sources say about making the voice as similar as possible (research only; no files in /home/user/voicetwin were edited)

## 0. Sources I used and what I could not reach

- **Official code.** I cloned github.com/RVC-Boss/GPT-SoVITS to `/tmp/claude-0/-home-user/3d87cc6c-41dd-58c2-802f-d51dc0c78527/scratchpad/gsv/repo`. That is main @6d51ffc (2026-11-12), plus the `cuda_graph_accel_v5` and `main20261003` branches and the last 30 main commits.
- **Which code the official package actually contains.** I matched file sizes from the package's file list (`/home/user/voicetwin/research/gsv_v4_v5/官方资料/v5整合包/文件清单.tsv`) against the repo history:
  - `TTS.py` (90195 B) and `inference_webui.py` (59029 B) are from commit **04a44eb** (2026-10-31, "Sync 20261031 v5 code update").
  - `models_v5.py`, `models.py` and `s2_train_v3_lora.py` are from **3b95794** (2026-10-06). Between these two commits the only differences in `models_v5.py` / `models.py` are a `generator=` argument; behaviour is the same.
  - The package's `TTS.py` is saved at `/tmp/claude-0/-home-user/3d87cc6c-41dd-58c2-802f-d51dc0c78527/scratchpad/gsv/TTS_pkg.py`.
- **Official wiki "GPT‐SoVITS‐features(Latest‐Including‐v5)".** My fetch was redirected to the README, so I could not read the live page. I used the full text the teacher pasted on 10-01/10-02, saved in `/home/user/voicetwin/research/过程原始记录/对话全文.md` lines 2395–2610. The page may have changed since then.
- **Not reachable:**
  - yuque (proxy 403), huggingface.co (403).
  - GitHub issues, PRs and the wiki API: 404 or not enabled for this session. The issues page only shows #1, so I found **no V5 issue or discussion threads**.
  - WebSearch returned nothing V5-specific.

## 1. Most important findings

**F1. Fine-tuning V5 is not officially supported. V5's official strength is zero-shot (no training).**
- Wiki (对话全文.md ~L2450): "V4版本的SoVITS可供用户微调，微调场景下音色相似度和V5不会有多少区别；V5版本的SoVITS，Zero Shot TTS音色相似度最高……**基于v5版本的少量样本微调暂未进行研究和实验**。" The version table marks only v4 as "可供微调".
- Commit 04a44eb (2026-10-31) **removed v5dev and v5turbo from the training-version choices**. `webui.py` L1501 now offers only `["v2","v2Pro","v2ProPlus","v4"]`. The package we analysed (re-uploaded to Hugging Face 2026-10-31) has the same list (`官方资料/v5整合包/webui.py` L1499).
- `s2_train_v3_lora.py` still trains V5 when called directly. VoiceTwin's CI run did exactly this (`research/gsv_v4_v5/真实V5实测/结果/日志/voicetwin/gsv_s2_train_v5dev.log`).
- Conclusion: V5 LoRA is technically possible but officially untested, and was withdrawn from the official training screen.
- **Recommendation:** add **"V5 base (no SoVITS training) + the teacher's fine-tuned GPT"** as a candidate in automatic checkpoint selection, alongside every V5 LoRA epoch. Today `select.py` / `plan_lora` have no base-model candidate. Measurements on held-out recordings then decide whether LoRA helps.
- Optional control candidate the wiki explicitly endorses: V4 LoRA + the V5 vocoder. The wiki says the two vocoders "二者可混用，v5声码器效果无脑高于v4" (~L2585). Keeping it as a hidden control fits the teacher's "V5 only" wish.

**F2. What "V5 LoRA" actually trains** (official code; numbers computed from it):
- LoRA covers **only the DiT attention layers** `to_q`, `to_k`, `to_v`, `to_out.0`, with `lora_alpha = rank`, so the scaling factor is 1 (`s2_train_v3_lora.py` L154–158). It is applied only to `net_g.cfm` (L188 / L218).
- Some other parts are **not frozen** and are fully trained: `ref_enc` (style encoder), `bridge`, `wns1`. Only `ssl_proj`, `quantizer` and `enc_p` are frozen (`models.py` L1323–1326). The DiT feed-forward and norm layers stay frozen.
- The DiT is dim 1024, 22 blocks, 16 heads (`models.py` L1316). LoRA size = 180,224 × rank: **rank 16 ≈ 2.9 M, 32 ≈ 5.8 M, 64 ≈ 11.5 M, 128 ≈ 23.1 M** parameters. The whole model is "330M+77M" per the wiki table.
- Optimizer: AdamW, lr 1e-4 (`configs/s2.json`), ExponentialLR with gamma 0.999875 per epoch (almost no decay), fp16, no gradient clipping (`clip_grad_value_(…, None)`).
- **The V5 training loss has no condition dropout** (`CFMV5.forward`, `models_v5.py` L31–46). The "no reference" branch that v5dev's CFG 1.30 uses at inference (`drop_audio_cond=True`, `models.py` ~L1171–1191) is therefore never trained during LoRA. How CFG behaves on a fine-tuned V5 is unknown. This is my inference from the code; untested.

**F3. Rank, epochs and batch: what the official sources actually say**
- **Rank.** Choices are 16/32/64/128, default 32 (`webui.py` ~L1736–1742). The official sources give no statement or data on how rank affects quality.
  - Judgement, from theory, untested for V5: with alpha = rank (scaling 1) and a fixed learning rate, the rsLoRA analysis (Kalajdzievski 2023, arXiv:2312.03732) says the stable scaling is proportional to 1/√rank. Rank 128 would then take roughly 2× larger effective steps than rank 32, so it may overfit sooner per epoch.
- **Epochs.** For v3/v4/v5 the default is 2, maximum 16, save every 1. The only reason given is the code comment "训太多=作死" (`webui.py` L129–131). Officially there are no overfitting signs or numbers.
  - Judgement: the training loss says little here. Use held-out speaker similarity, character error rate, and listening for artefacts or "电音" (the UI's step-count hint says to raise steps "如果觉得电").
  - The data loader repeats data to at least 100 items (`data_utils.py` L547–551). With about 1,000 clips and batch 2, one epoch is about 500 steps (computed).
- **Batch.** Official rule is `vram_gb // 8` for v3/v4/v5 versus `// 2` for v2 (`webui.py` L119). This implies about 8 GB per sample in a batch; that is derived from the rule, not measured. Gradient checkpointing (`grad_ckpt`) saves memory. Without it, batch size stays at `vram_gb // 8`. On the teacher's RTX 50 that means batch 2 on a 16 GB card, 4 on a 32 GB card.
- **What training keeps.** The loader keeps clips from 0.6 to 54 s (`data_utils.py` L96/L350). The LoRA length buckets `[32 … 1000]` at 50 frames/s **drop clips outside 0.64–20 s** (`s2_train_v3_lora.py` L111–121; the bucket sampler discards anything outside the boundaries, `data_utils.py` L983). VoiceTwin clips are 2–12 s (`prepare.py` L356), so all of her material is used.

**F4. Full (non-LoRA) fine-tuning of the V5 SoVITS is not supported**
- `s2_train_v3.py` (full fine-tune) is hard-wired to the v3 loader (24 kHz, hop 256; L34) and is not called by the web UI.
- `process_ckpt.py` has version headers only for V5 LoRA ("07" = v5dev-lora, "08" = v5turbo-lora).
- Inference hard-codes the LoRA target modules (`TTS.py` L656–667).
- Full fine-tuning or wider LoRA targets (for example the feed-forward layers) would need custom code in both training and loading. Untested; high risk.

**F5. v5dev vs v5turbo**
- Wiki SeedTTS Chinese zero-shot results (对话全文.md L2466–2471), similarity score / error rate:
  - v5dev, 32 steps, CFG 1.3: **0.805** / 0.017
  - v5turbo, 4 steps: 0.782 / 0.016
  - v5turbo, 8 steps: 0.783
  - v4, 8 steps: 0.735
  - real speech (GT): 0.755
- Wiki: "Dev版本仅供刷榜和基模研究用，32步cfg推理太慢了不建议普通用户使用；Turbo版本为dev版本的4步nocfg DMD蒸馏". The wiki also says error rate depends on the GPT, not on SoVITS.
- **For quality, use v5dev.** Each generation window costs 32 steps × 2 (CFG) = 64 DiT passes, versus 4 for turbo: 16× more DiT compute (computed from code; wall-clock time untested).
- v5dev is a plain flow-matching model: `use_step_embedding` is False for V5 (`models.py` L1317), and V5 has no shortcut training (the v3/v4 `CFM.forward` has a 30% shortcut branch, L1209–1222). Using fewer than 32 steps on dev is not what it was designed for.
- **Do not LoRA-train v5turbo.** Judgement from code: LoRA uses the same flow-matching loss (`CFMV5.forward`) on a model that was DMD-distilled for 4 steps, which may break the distillation.
- README chart (`官方资料/零样本相似度对比图.png`): "GPT-SoVITS v5pro（预训练进行中）0.410B · 0.824+". Wiki: versions after v5 "may" need the speaker-verification model, as v2Pro does. Worth watching.

**F6. Two official front ends run V5 differently. VoiceTwin's path matches the official benchmark.**
- `TTS.py` (behind `api_v2`, which VoiceTwin uses) runs `synthesize_v5_mel`:
  - defaults: steps 32 (dev) / 4 (turbo), CFG 1.30 (dev) / 0 (turbo) (`models_v5.py` L53–54; `TTS.py` L1203–1204);
  - fixed noise temperature 0.875 (`models_v5.py` L7, L22);
  - V5 vocoder (`TTS.py` ~L746–750).
- The classic `inference_webui.py` after 04a44eb sends V5 through the v4 path: CFG 0, temperature 1.0, default 8 steps, and the **v4** vocoder (04a44eb diff).
- So do not copy settings from the classic web UI. The package `api_v2` accepts `cfg_rate` and `sample_steps` (`官方资料/v5整合包/api_v2.py` L180–181, L483–513). VoiceTwin currently sends neither, so it already gets the dev defaults of 32 steps and CFG 1.3.

**F7. How V5 uses the reference recording, and why long sentences are split** (code; numbers computed)
- The reference must be 3–10 s, otherwise it is rejected (`TTS_pkg.py` L918). Only the first reference is used (L1747); extra references do nothing. There is no mode without reference text (L1222).
- The GPT prompt is the **whole** reference plus 0.3 s of padding (L908–931). The style vector `ge` comes from the whole reference.
- The SoVITS acoustic prompt is only **the last 500 mel frames = last 5.0 s** of the reference (`models_v5.py` L11, L56–58; mel runs at 100 frames/s, 32 kHz with hop 320, `TTS.py` L159–170).
- Generation window = min(1000 − reference frames, 640):
  - with a reference ≥ 5 s, the window is **5.0 s**;
  - with a reference of 3.6 s or less, the window is 6.4 s.
- Later windows use the reference prefix plus the last 0.32 s of already-generated audio (`models_v5.py` L59–82). Any sentence longer than about 5 s is therefore rendered in several acoustic windows. The GPT token sequence stays continuous across them.
- Recommendations:
  - Prefer references of 5 s or more whose **last 5 s** is clean, representative speech that does not end in a long silence.
  - Measure whether long sentences show seams at the window boundaries. No official data exists; untested.

**F8. Speaking habits (语气, rhythm, pauses) are learned by the GPT, and steered by the reference**
- The SoVITS output length is fixed by the GPT's token count: 4 frames per token at 25 Hz, divided by speed (`models.py` `decode_encp`, ~L1366–1371). So **timing, rhythm and pauses come entirely from the GPT**.
- Wiki: the GPT decides stability and error rate, and V2Pro(Plus)/V3/V4/V5 share the same GPT, which "更容易出丰富、拟真的情感".
- README L353: "V3/V4 合成的声音更偏向参考音频，而不是整体训练集的风格". **Choosing a matching reference for each sentence is therefore a main lever for V5** (judgement, applying that statement to V5's diffusion route).
- **GPT training:**
  - `s1longer-v2.yaml`: default 15 epochs, maximum 50 (`webui.py` UI ~L1799–1807), batch `vram // 2` (L120).
  - The learning rate is in effect **locked at 0.002** (`AR/modules/lr_schedulers.py` L58: "不听话，直接锁定！"); warmup and decay settings are ignored.
  - DPO is labelled "实验性". Its rejected examples are synthetic repeated or dropped token spans (`AR/models/utils.py` L244–282). It targets fewer repeats and skips, **not** prosody.

**F9. 48 kHz output, but the teacher's sound is learned only up to 16 kHz**
- The acoustic model works on a 100-band mel at 32 kHz (top frequency 16 kHz) (`TTS.py` L159–170; `data_utils.py` V4 loader). The vocoder upsamples ×480 to 48 kHz (`TTS.py` L735–757).
- So anything above 16 kHz in the output comes from the stock vocoder, not from the teacher's data. Nobody fine-tunes the vocoder, and there is no official script for it.

**F10. cache-dit (speed-up)**
- Off by default. It exists only in main @59c4468 (2026-11-09), **not in the package**.
- Commit message measurement on v5dev, 32 steps: mel relative L1 error 1.2e-3 versus no cache, DiT about 1.12× faster.
- Keep it off for maximum fidelity. It does not affect silence.

## 2. Answers to the questions

**(1) Training levers**
- Rank, epochs, batch, learning rate: as in F2/F3. Official settings: rank 32, 2 epochs (max 16), lr 1e-4, batch `vram // 8`.
- The official sources have no data on how rank or epoch count affects V5 quality, because V5 few-shot is "未研究".
- Full fine-tuning of V5 SoVITS: no (F4).
- GPT: 15 epochs default, max 50, lr locked 0.002; DPO is for stability only (F8).
- **Practical plan (untested):**
  - Train v5dev LoRA at rank 32 and 64 (128 optional), saving every epoch from 1 to 8.
  - Keep the official e1/e2 checkpoints, and the V5 base model, as candidates.
  - Go up to 16 epochs only if the held-out scores are still rising at 8.
  - Train the GPT on all of her clips. Sweep about 10–25 epochs and choose by held-out duration, pause and pitch statistics as well as speaker similarity and error rate.

**(2) Inference levers**
- `sample_steps`: 32 for dev (the official benchmark setting); higher is untested.
- `cfg_rate`: default 1.30; UI range 0–2 in steps of 0.05 (`inference_webui_fast.py` ~L383–386). It is a "follow the reference" knob, because the negative branch drops the reference audio. **After LoRA, sweep {0, 0.7, 1.0, 1.3, 1.6}** (F2).
- GPT sampling: top_k 15, top_p 1, temperature 1, repetition_penalty 1.35 are the defaults (`TTS.py` docstring ~L1146–1158). There is no V5-specific advice; they act on the GPT only.
- The acoustic noise temperature is fixed at 0.875 and cannot be set through `api_v2`.
- Reference: 3–10 s, use at least 5 s, last 5 s matters, accurate reference text required, only one reference counts (F7).
- `speed_factor`: stretches the content features by interpolation (`decode_encp`). No official quality data. Judgement: keep it near 1.0 once the GPT has learned her tempo.
- Text splitting: VoiceTwin's per-sentence `cut0` is fine. The official fast UI defaults to "凑四句一切". V5's acoustic windowing does not depend on text splitting.
- Parallel / batch: for V5, each item in a batch goes through `using_vocoder_synthesis` on its own (`TTS_pkg.py` L1825–1833), so batching does not change the acoustic path. The fast UI sets V5 batch to 1 (`inference_webui_fast.py` ~L416). Expected to be quality-neutral; untested.
- `super_sampling` is v3 only.

**(3) Known V5 weaknesses and official workarounds**
- Official statements:
  - dev is slow;
  - few-shot V5 is not studied (workaround: V4 LoRA, or V5 zero-shot);
  - the diffusion route needs high-quality training audio (the teacher's studio audio fits);
  - output follows the reference rather than the training set's style;
  - no mode without reference text; 3–10 s reference; auxiliary references ignored.
- English and mixed Chinese-English: **no official V5 numbers**. The SeedTTS table is Chinese only, so measure per language group with `eval/lang_groups.py`.
- Long sentences: the 5–6.4 s window joins (F7), found from code; no official note.
- Reference carry-over: diffusion is "更忠实于参考音频". The reference's microphone, room, noise floor and emotion carry into the output. Use clean teacher-only references; this also protects the absolute-silence requirement. Pauses inside a sentence are model output, not digital zero, so measure their noise floor on her PC.

**(4) dev vs turbo:** see F5. For quality: v5dev, 32 steps, CFG 1.3, V5 vocoder, through `api_v2`.

## 3. How to verify

**On GitHub Actions (CPU only), plumbing checks:**
- `cfg_rate` and `sample_steps` reach the API;
- rank 16/32/64/128 LoRA checkpoints load (header "07");
- count the acoustic windows produced for a sentence longer than 5 s, for references of 3.5 s and 5+ s;
- the base model shows up as a selection candidate.

**On the teacher's PC (real quality):**
- Compare candidates {V5 base, LoRA rank 32/64 × e1…e8} × CFG {1.0, 1.3, 1.6}.
- Use her held-out sentences, including pure Chinese, mixed Chinese-English and long sentences.
- Metrics: VoiceTwin's three-model speaker similarity %, ASR error rate, duration/pause/pitch distributions against her real recordings, pause noise floor, and window-boundary artefacts.
- She makes the final call by blind listening.
- Training time and memory on RTX 50: unmeasured; record them in that run.

**Licence:** the GPT-SoVITS code is MIT (repo `LICENSE`). I did not check the pretrained-model licence.

