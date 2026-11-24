# 调研：new-strategies（2026-11-24）

REPORT: Training and inference strategies from 2025 to Nov 2026 for making the fine-tuned GPT-SoVITS V5 voice more like the teacher and more human, and what applies to V5

Access limits. arxiv.org, huggingface.co, aclanthology.org and openreview were blocked by the session proxy (HTTP 403 / DNS fail). Paper numbers below therefore come from search-engine extracts of each paper's abstract or HTML. I did not read the full PDFs, and every such number is tagged [abs]. Facts tagged [code] I checked myself in raw GitHub source (raw.githubusercontent.com works). Anything tagged [unverified] is my own inference. Nothing under /home/user/voicetwin was edited. My code copies are in /tmp/claude-0/-home-user/3d87cc6c-41dd-58c2-802f-d51dc0c78527/scratchpad/ (models.py, dit.py, TTS.py, api_v2.py, lora.py, t2s_model.py, arutils.py).

==================================================================
0. BOTTOM LINE
==================================================================
- Best-of-N candidate picking is already the main test-time scaling method in the field, and VoiceTwin already uses it heavily (up to 64 candidates, 3 SV models, CER, pitch, rate, continuity). Published RL and preference methods mostly cut content errors. Their speaker-similarity gains are small (about +0.3 to +2.3 SIM points in zero-shot settings), and several papers report flatter prosody or unstable SIM. They would mainly cut the candidates needed, not raise the ceiling.
- The cheapest real gains are inference knobs that VoiceTwin leaves at official defaults:
  - V5's classifier-free guidance strength (`cfg_rate`, default 1.30)
  - a guidance interval (apply CFG only on part of the sampling steps)
  - "autoguidance" against an early or low-rank LoRA checkpoint
  - power-of-two step counts
  All of these are measurable with the existing held-out scoring.
- The biggest risk at N=64 is "verifier hacking": picking outliers that fool the SV models. Hold one SV model out of selection and use it only to audit.
- For human-likeness, the weak points are pauses that ignore sentence length, paragraph-level prosody resets, and transcripts that drop fillers. Each has a cheap fix. Absolute digital silence is NOT a known human tell, and the teacher says her own recordings have silent gaps, so keep the rule.
- Two well-founded training experiments, each accepted only if the held-out score improves:
  1. SpeechAlign-style DPO on the GPT stage: her real semantic tokens as "chosen", the model's own samples as "rejected".
  2. Rewards computed from VoiceTwin's existing scored candidates, used either for Koel-style DPO pairs or for GROW-style reward-weighted flow-matching LoRA training.

==================================================================
1. WHAT V5 EXPOSES (code-verified, RVC-Boss/GPT-SoVITS main)
==================================================================
- V5 SoVITS is a DiT flow-matching model (CFM) with classifier-free guidance [code: GPT_SoVITS/module/models.py, class CFM.inference]:
  - the guidance formula is `v = v + (v - v_uncond) * inference_cfg_rate`
  - the unconditional branch drops both the reference mel and the semantic features
- `cfg_rate` defaults [code: TTS_infer_pack/TTS.py]:
  - v5dev: 1.30
  - v5turbo: 0.0
  - api_v2.py accepts `cfg_rate` per request, so it needs no patch.
- `sample_steps` defaults: v5dev 32, v5turbo 4 [code].
- Step embedding [code: CFM.forward]:
  - The CFM is trained shortcut-style: the step size `dt` is an input.
  - Trained step sizes are 0 and 1/2, 1/4, ... 1/64.
  - [unverified, inferred from the code] So 8/16/32/64 steps are "on-distribution". Non-power-of-two counts and 128 are not.
- cache-dit (DBCache) [code]: off by default; the code comment says "near-identical audio" with fewer DiT passes. It could buy more candidates per second. Verify with your own scoring.
- The CFM has a `noise_temperature` attribute [code], not exposed in the API.
- LoRA setup [code: s2_train_v3_lora.py]:
  - LoRA covers only the DiT attention layers (`to_q`, `to_k`, `to_v`, `to_out.0`), with `lora_alpha = r` (scale 1).
  - The fine-tune loss (CFM.forward) has NO condition dropout; the drop flags in dit.py are marked "###no-use".
  - So after fine-tuning, the unconditional branch is never trained on her data, and the official 1.30 was tuned for the base model. [unverified] The best `cfg_rate` for her LoRA may differ.
- The GPT "DPO" option [code: AR/models/utils.py make_reject_y; Changelog 2024-02-12]:
  - It builds rejected samples by repeating or deleting spans of the real semantic tokens (reference-free DPO, beta 0.2).
  - The official purpose is "缓解 GPT 重复漏字问题" (reduce repeated or skipped words). It does not target similarity, which matches VoiceTwin's current "off" default.
- A full (non-LoRA) SoVITS fine-tune script exists (`s2_train_v3.py`) [code], but webui.py routes v3/v4/v5 to LoRA. Full fine-tuning of V5 is untested.

==================================================================
2. RANKED RECOMMENDATIONS
==================================================================
Acceptance test for every item: the existing held-out set (20 validation clips + 24 test texts).
- Require: calibrated "像你本人%" up, CER not worse, and the prosody-distribution distance to her recordings not worse.
- Also require an AUDIT SV model that is never used for selection to move the same way (see A4).
- Keep a change only if all of these hold.

--- Tier A: inference only, no retraining, cheap (do first) ---

A1. Tune `cfg_rate` on HER LoRA model; optionally make it a candidate axis.
- Plug-in: api_v2 `cfg_rate`, no patch. Sweep 0 / 0.7 / 1.0 / 1.3 / 1.6 / 2.0 / 2.5.
- Evidence:
  - Koel-TTS (NVIDIA, arXiv 2502.05236, EMNLP 2025): raising CFG scale lowers CER and raises SSIM; CFG and preference alignment add up. [abs]
  - Align2Speak re-ran Koel 380M [abs, arXiv 2509.21718]: base CER 2.68% without CFG vs 0.57% with CFG; SSIM base+CFG 0.720 vs DPO+CFG 0.729.
  - BareWave (arXiv 2606.09048): SIM-o peaks at CFG 3.5, then falls. [abs]
  - Selective CFG (arXiv 2509.19668): speaker-emphasised guidance helped English F5-TTS but did NOT help Mandarin F5-TTS. Do not expect speaker-only guidance tricks to transfer to Chinese. [abs]
- Windows: trivial.

A2. Guidance interval (apply CFG only on part of the steps, not at pure noise).
- Plug-in: about a 3-line monkeypatch in CFM.inference (only compute the negative pass when `t >= t0`). This also saves compute.
- Evidence:
  - Kynkäänniemi et al., NeurIPS 2024 (arXiv 2404.07724): guidance is harmful at high noise and useful mid-chain; they recommend the interval as a hyperparameter. [abs]
  - "Guidance at Pure Noise Erodes Speaker Similarity in Few-Step Token-Conditioned Flow-Matching TTS" (arXiv 2611.11179, Nov 2026) [abs]:
    - Setting: token-conditioned flow-matching TTS, the same class as V5. With 2 steps, guiding the first (pure-noise) evaluation costs 1 to 6 SIM points at unchanged WER.
    - Their fix (first evaluation unguided, one midpoint step) recovers 77 to 85% on IndexTTS-2.
    - This matters most if v5turbo (4 steps) is ever used.
  - Joint Residual Reweighting (arXiv 2606.25672): applying the correction at only a few steps keeps most of the gain. [abs]
- Sweep t0 in {0, 0.1, 0.2, 0.3}.

A3. Autoguidance: guide the final LoRA away from a weak version of itself.
- Formula: `v = v_good + w*(v_good - v_weak)`, both branches fully conditioned.
- Evidence: VoiceGuider (arXiv 2409.15759, ICASSP 2025) [abs]:
  - The weak model was a rank-1 LoRA trained for 100 iterations.
  - It brought a LoRA speaker-adapted diffusion TTS to the SECS of full fine-tuning; on full-FT models the gain was minor.
  - The idea comes from Karras et al. 2024, "Guiding a diffusion model with a bad version of itself".
- Plug-in for V5:
  - Weak model = the epoch-1 LoRA checkpoint VoiceTwin already saves, or a quick rank-1/4 LoRA.
  - Needs a patch: a second peft adapter plus `set_adapter` swap for the negative pass. It replaces or blends with the CFG pass, so cost is about the same as CFG.
- [unverified on V5] Evidence is out-of-domain and zero/one-shot. Gain on an in-domain 2.5 h fine-tune is unknown.

A4. Guard best-of-N against verifier hacking.
- Evidence:
  - Inference-time scaling for diffusion audio super-resolution (arXiv 2508.02391): speaker-similarity and aesthetics verifiers gave diminishing or negative returns on other metrics beyond about 2^3 to 2^4 candidates ("verifier hacking"). [abs]
  - The LLM best-of-N literature (e.g. regularized BoN, arXiv 2404.01054) shows quality degrading as N grows under an imperfect reward.
- Action:
  - Keep one SV model out of selection, used only as an audit. Example: the WavLM-large SV model used by Seed-TTS-eval, or CAM++, which VoiceTwin already lists as a backup.
  - Log audit score vs N per sentence. If the audit plateaus at N≈8 to 16 while the selection score keeps rising, cap N there. That also saves time.
  - Prefer candidates near her real p50 on prosody features over max-SV outliers (VoiceTwin's p50 search target already leans this way).

A5. Naturalness rerankers: what works and what does not.
- Do NOT add UTMOS/UTMOSv2 or DNSMOS as within-system rerankers.
  - TTSDS2 (arXiv 2506.19441, ICLR 2026) [abs]: UTMOS averaged about -0.04 Spearman across its domains; TTSDS2 was the only one of 16 metrics above 0.5 in every domain.
  - A 2026 study reports UTMOSv2 run-to-run instability. [abs]
- Speaker SV vs human judgment [abs]:
  - DMOSpeech (arXiv 2410.11097): per-utterance WavLM-SIM vs SMOS r = 0.55 (voice) and 0.50 (style).
  - So SV captures timbre, not style. Keep VoiceTwin's prosody terms.
- The best published pairwise naturalness judge is SpeechJudge-GRM (arXiv 2511.07931, Nov 2025, ICLR 2026) [abs]:
  - 77.2% agreement with human A/B picks (79.4% with @10 inference-time scaling), vs 72.7% for a Bradley-Terry RM; Gemini-2.5-Flash scored under 70%.
  - Built on Qwen2.5-Omni-7B; code at github.com/AmphionTeam/SpeechJudge, weights at RMSnow/SpeechJudge-GRM.
  - Licence: no LICENSE file at the repo root [code: 404]; unverified.
  - Language coverage: "multiple languages"; Chinese coverage unverified.
  - VRAM [unverified estimate]: 7B in bf16 is at least about 14 GB for weights alone, so it cannot share the GPU with GPT-SoVITS. vLLM is Linux-only, but the transformers path should work on Windows.
  - Use: an optional final tie-breaker between the top 2 candidates of sentences whose scores are within noise. Not for every candidate.

A6. Step count and speed.
- Try 64 vs 32 steps (both trained step sizes [code-inferred]).
- Try cache-dit at 32 steps to buy more candidates per minute.
- Pick by the A4 audit, not by the selection score.
- Skip F5-style sway sampling unless measured. F5-TTS reported SIM 0.62→0.66 at 32 steps (arXiv 2410.06885, appendix) [abs], but V5's step-size embedding makes non-uniform steps an untested change [unverified].

A7. Not applicable to V5 (save the effort).
- Prompt-Consistency Inference (Microsoft, arXiv 2610.04757) resets the prompt region of the ODE state. V5 instead zeroes that region during both training and inference and feeds the prompt as a separate condition [code], so PCI does not transfer.
- F5R-TTS GRPO (arXiv 2504.02407, MIT code at github.com/FrontierLabs/F5R-TTS) needs a probabilistically re-pretrained model [code README], so it does not transfer either.

--- Tier B: data curation (re-train once) ---

B1. Verbatim transcripts.
- Whisper-family ASR drops fillers and repairs (CrisperWhisper paper, arXiv 2408.16589) [abs]. Audio-text mismatch teaches the GPT stage to misalign or hallucinate.
- Evidence:
  - Székely et al. 2019: precise filled-pause annotation improved rated naturalness.
  - Matsunaga et al. (arXiv 2210.09815): inserting fillers can LOWER naturalness. [abs]
  - So: annotate what she actually says; do not add fillers to scripts.
- Tool: CrisperWhisper 2.0 "verbatimize" (clean transcript + audio → adds only the disfluencies actually present) [code: README].
  - Licence: non-commercial research only.
  - Its 10-language disfluency benchmark is vendor-run; Chinese coverage unverified.
  - Alternative: compare the two ASR engines VoiceTwin already has and flag segments whose syllable rate is implausible.

B2. Find transcript mismatches with the model's own loss.
- Analogue of Self-Purifying Flow Matching (arXiv 2509.19091; WildSpoof 2026, arXiv 2512.17293): flag pairs whose conditional loss is not clearly below the unconditional loss, and route them out or to unconditional training. [abs]
- [unverified adaptation] For GPT-SoVITS, use the per-utterance GPT NLL of the real semantic tokens: high outliers are likely transcript mismatches. Review or drop them before the final training.

B3. Quality filter.
- DNSMOS 2.8 was the best threshold for SIM-o in flow-matching pre-training (arXiv 2406.05699) [abs].
- Her studio recordings probably pass anyway. Filter only clear outliers.
- Data diversity matters more (arXiv 2603.10904, LoRA TTS) [abs]: keep questions, emphasis, and English-containing segments in training.

--- Tier C: training experiments (higher cost, uncertain; one at a time, A/B vs current) ---

C1. SpeechAlign-style DPO on the GPT stage (label-free).
- Chosen = her real semantic tokens; rejected = the fine-tuned GPT's own samples for the same text.
- Evidence: SpeechAlign (NeurIPS 2024, arXiv 2404.05600) beat SFT on LibriSpeech/VCTK [abs].
- Plug-in: GPT-SoVITS already has `dpo_loss` and `get_batch_logps` [code]. Only `make_reject_y` needs to be replaced by loading pre-generated token sequences.
- VRAM: DPO halves the batch (official changelog).
- Caveat: "Reliable Neural-Codec TTS ..." (arXiv 2606.18323, Jun 2026) found offline DPO/IPO did NOT beat plain supervised distillation of best-of-N outputs for robustness [abs].

C2. Use VoiceTwin's existing scored candidates as reward data.
- (a) Koel-style DPO pairs for the GPT stage, chosen = high combined score, rejected = low.
  - Evidence [abs]: Koel-TTS gains; CosyVoice 2's DPO helped normal zh/en but hurt hard cases (per arXiv 2509.18569 summary).
- (b) GROW-style reward-weighted flow-matching LoRA training of the V5 CFM (arXiv 2608.03215, Aug 2026) [abs]:
  - Flow-matching regression on generated samples, weighted by within-group standardized SIM and intelligibility rewards, with a Wasserstein-2 anchor to the frozen model.
  - Needs only scalar rewards, so VoiceTwin's CPU SV/ASR scorers suffice.
  - Fits into s2_train_v3_lora.py's loss.
  - Related: LoCA (arXiv 2610.28035, Oct 2026) adds frame-level credit from ASR alignment: on VoxCPM2 it cut content error 20.5% vs the base model and 14.3% vs GROW, and also raised SIM [abs]. FlowTTS-GRPO (arXiv 2606.23190): online RL on CosyVoice3/F5-TTS [abs only, details unverified].
- Expected size of gains:
  - GLM-TTS GRPO [code README, Seed-TTS-eval zh]: CER 1.03→0.89, SIM 76.1→76.4.
  - F5R internal zh set [abs]: SIM 0.731→0.754, WER 1.68→1.37.
  - So mainly fewer failures, which means fewer candidates needed.
- Reference implementation: GLM-TTS grpo/ (Apache-2.0) [code]. It uses SIM, CER, emotion and laughter rewards; its pitch/energy-variance rewards are zeroed in the released code.
- Mandatory guards:
  - "No Verifiable Reward for Prosody" (arXiv 2509.18531): GRPO on CER/NLL collapsed prosody toward monotone; adding a SIM reward destabilised training. Their fix was iterative DPO on about 200 human preference pairs per round [abs].
  - Multi-Reward GRPO (arXiv 2511.21270) adds length, entropy and prosody rewards; the prosody reward had the largest MOS effect [abs].
  - So: include a "distance to her prosody distribution" term and reject any run whose F0/rate variance shrinks.
- Windows: possible in the 整合包's Python. Expect days of engineering; GPU time scales with group size.

C3. LoRA rank / full fine-tuning: low priority.
- VoiceTailor (Interspeech 2024): even rank 2 is nearly enough; adding non-attention layers did not help SIM [abs].
- VoiceGuider: LoRA about 0.90 vs full fine-tuning 0.904 SECS [abs].
- Rank 64/128 is worth one sweep at most, and only via VoiceTwin's automatic checkpoint selection.
- Adding F5-style condition dropout to the LoRA fine-tune (F5-TTS uses audio_drop_prob=0.3, cond_drop_prob=0.2 [code: f5_tts/model/cfm.py]) would keep CFG calibrated on her model [unverified for V5]. Try it only if A1 shows that CFG behaves oddly after fine-tuning.

--- Tier D: human-likeness fixes at the assembly level ---

D1. Pause length should depend on the NEXT sentence.
- Today `_twin_pause` samples her comma/sentence quantiles at u∈[0.2,0.8] independently of context [code: voicetwin/synth/engine.py:2646]. This truncates her tails and ignores planning effects.
- Evidence:
  - Krivokapić 2007, J. Phonetics 35(2): pause duration grows with upcoming phrase length.
  - Fuchs et al. 2013, J. Phonetics 41(1): pause and inhalation duration differ with sentence length, not with syntactic complexity.
  - Stochastic-duration TTS work: identical timings are unnatural.
- Fix: fit pause ~ kind + length(next sentence) (+ paragraph position) on her recordings, then sample the residual from her full empirical distribution. Apply only if the fitted slope is significant in her data.

D2. Paragraph-level prosody continuity.
- Per-sentence generation resets pitch and energy.
- Evidence [abs]: bounded preceding-context conditioning helps.
  - Context-aware memory, arXiv 2508.14713.
  - Bounded context, arXiv 2603.06444: unbounded history collapses.
  - Amazon multi-sentence TTS, arXiv 2206.14643.
  - NVIDIA Magpie-TTS long-form prepends previous chunks.
- V5 plug-in [unverified]: offer the previous chosen sentence as the GPT prompt (text plus semantic tokens) while keeping HER real reference for the CFM mel prompt.
  - api_v2 uses one reference for both, so this needs a small patch; otherwise timbre drifts.
- VoiceTwin's whole-lecture continuity re-pick already partly covers this.

D3. Breaths.
- Evidence that breaths help perceived naturalness:
  - Yang et al. 2024 (arXiv 2402.00288): 40-listener MOS gain from breath marks.
  - Elmers et al. 2021 and 2023.
  - Listeners cite breathing as a cue: San Segundo et al., arXiv 2512.09221, 54 listeners, 59.11% accuracy.
- Conflict: anything audible inside a gap breaks the absolute-silence rule.
  - VoiceTwin already records her breath fraction (twin_profile.py; "只记录", record only).
  - Only if her recordings show onset breaths, and only with her approval: train with breath marks so breaths are generated attached to speech onsets, not inserted into gaps. Otherwise skip.

D4. Silence.
- The only published silence "tell" is the distribution of leading/trailing silence DURATION, a shortcut learned by ASVspoof detectors (Müller et al. 2021, arXiv 2106.12914: EER 3.6%→15.5% after trimming) [abs].
- Digital zero is not a documented human cue, and her own recordings reportedly have silent gaps (research/gsv_v4_v5/记录.md).
- Keep absolute silence; match her lead/trail and gap duration distributions (D1).
- Loudness: VoiceTwin already preserves her sentence-to-sentence variation (engine.py `_levels`). No change.

==================================================================
3. HOW TO PROVE "LISTENERS CANNOT TELL"
==================================================================
Automatic metrics correlate only moderately with humans (TTSDS2; DMOSpeech r≈0.5), so the final check has to be a human test.

Proposed blind test (design is mine):
- Use her held-out real sentences vs generated versions of the same texts, including 30 to 60 s paragraph clips.
- Listeners who know her voice; real/AI judgement.
- Success = accuracy not significantly above 50% (binomial CI).

Published human baselines:
- About 60% detection for ElevenLabs clones, and identity judged "same" about 80% of the time (Barrington, Cooper & Farid, Sci. Rep. 2025; the 80% is from the preprint).
- 59.11% in San Segundo et al. 2025.

Corpus-level audit:
- TTSDS2 (`pip install ttsds`, MIT, v2.1.3 [code README]) can score a generated lecture against her real recordings factor by factor (prosody, speaker, intelligibility). Use it to tune knobs (A1 to A3), not to pick candidates.
- Windows is unverified: it lists apt dependencies and fairseq, so it is likely Linux/WSL.
- Her audio is private, so this cannot run on the public repo's GitHub Actions.

==================================================================
4. TOOL / METHOD SHEET
==================================================================
SpeechJudge-GRM
- Date: Nov 2025. Licence: not stated. zh/en: multiple languages, zh unverified.
- Numbers: 77.2% / 79.4% human agreement [abs].
- VRAM: about 14 GB+ [estimate].
- Windows: transformers path OK; vLLM no.
- Plug-in: final pairwise tie-break.

TTSDS2
- Date: Jun 2025 (ICLR 2026). Licence: MIT. zh/en: multilingual.
- Numbers: only metric of 16 above 0.5 Spearman in all domains [abs].
- Plug-in: corpus audit and knob tuning.

UTMOSv2
- Licence: MIT. VoiceMOS 2024 Track 1 winner [code README].
- Not recommended as a reranker (A5).

Audiobox-Aesthetics
- Date: 2025. Licence: CC-BY 4.0 [code].
- PQ/PC/CE/CU axes; no within-TTS evidence. Optional.

CrisperWhisper 2.0
- Licence: non-commercial.
- Disfluency F1 87.8 (vendor-reported); zh unverified.
- Plug-in: verbatim transcripts (B1).

GLM-TTS GRPO code
- Date: Dec 2025. Licence: Apache-2.0.
- Plug-in: template for C2 on the GPT stage.

F5R-TTS code
- Licence: MIT code; F5 weights CC-BY-NC.
- Not directly applicable to V5 (A7).

GROW / LoCA / FlowTTS-GRPO / FSU-Mid / Selective CFG / PCI / TLA-SA
- arXiv preprints from Sep 2025 to Nov 2026.
- Code availability not verified. Abstract-level evidence only.

Sources (all URLs as found):
- https://github.com/RVC-Boss/GPT-SoVITS
- arXiv: 2502.05236, 2509.21718, 2606.09048, 2509.19668, 2404.07724, 2611.11179, 2606.25672, 2409.15759, 2508.02391, 2404.01054, 2506.19441, 2410.11097, 2511.07931, 2410.06885, 2610.04757, 2504.02407, 2408.16589, 2210.09815, 2509.19091, 2512.17293, 2406.05699, 2603.10904, 2404.05600, 2606.18323, 2509.18569, 2608.03215, 2610.28035, 2606.23190, 2512.14291, 2509.18531, 2511.21270, 2402.00288, 2512.09221, 2106.12914, 2508.14713, 2603.06444, 2206.14643
- Paper and project pages: https://ojs.aaai.org/index.php/AAAI/article/view/40450, https://www.ischool.berkeley.edu/research/publications/2025/people-are-poorly-equipped-detect-ai-powered-voice-clones, https://www.phonetik.uni-muenchen.de/~hoole/pdf/fuchsetal_jphon_41_1_2013.pdf, https://dblp.uni-trier.de/rec/journals/jphonetics/Krivokapic07.html
- Repos: https://github.com/AmphionTeam/SpeechJudge, https://github.com/ttsds/ttsds, https://github.com/sarulab-speech/UTMOSv2, https://github.com/facebookresearch/audiobox-aesthetics, https://github.com/nyrahealth/CrisperWhisper, https://github.com/zai-org/GLM-TTS, https://github.com/FrontierLabs/F5R-TTS, https://github.com/SWivid/F5-TTS
