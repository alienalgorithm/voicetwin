# 调研：data-absorb（2026-11-24）

REPORT: getting V5 to fully absorb all of the teacher's material (research only; nothing in /home/user/voicetwin was changed)

Limits on what I could check
- www.yuque.com (the official 语雀 docs) and arxiv.org cannot be reached from this session because the proxy blocks them. Paper findings below therefore come from abstracts and search snippets, not full texts, and are marked as such.
- raw.githubusercontent.com and the GitHub wiki could be reached.
- Official code was fetched at commit 3b95794c (the "V5新版本" entry in Changelog_CN.md, 2026-10-05) and at main. The training-path files are md5-identical between the two: data_utils.py, s2_train_v3_lora.py, AR/data/dataset.py, text/chinese2.py, s1_train.py, 2-get-hubert-wav32k.py. Their byte sizes also match the teacher's package list (research/gsv_v4_v5/官方资料/v5整合包/文件清单.tsv), so the official line numbers below apply to her package.
- Everything labelled JUDGEMENT or UNTESTED is my opinion with no measurement behind it.

=== A. Main findings ===
1. Two different parts of V5 do the learning. The GPT (s1) predicts semantic tokens, so it carries her rhythm, pauses, intonation and habits (语气). The V5 SoVITS LoRA renders her timbre. At synthesis V5 only uses the last 5 s of the first reference (see B6). So "learn all of her habits" is mainly a question of GPT data and transcript accuracy; timbre is mainly the LoRA plus the reference.
2. The worst leak, the official 1A step deleting English inside Chinese lines, is already fixed by VoiceTwin's mixed text frontend. What remains is the fallback path (D9).
3. Several VoiceTwin prepare filters can silently throw away exactly the "habit" material: short interjections, expressive or slow lines, low-confidence code-switched lines, the new 26-min session (through the voiceprint filter), and real Bilibili phrases (through the hallucination list). None of these has been counted on her data. Each one is a `drop_reason` counter in her manifest, so measure on her PC first (section D).
4. Neither official trainer has a validation set. VoiceTwin's held-out selection is the only guard against under- and over-training. My suggestion is to add cheap held-out losses computed per checkpoint, and to extend training automatically while the last epoch is still the best (section J).
5. Her own polished transcripts (母本), measured: 0 occurrences of 嗯/呃/啊 in 1005 lines, 2 garbled English tokens, and 90 English passages of 6 or more words inside Chinese lines (section C).
6. Do not denoise studio material. The loudness handling is fine.

=== B. What the official V5 training code does with each clip ===
B1. Which clips the V5 LoRA actually trains on
- The V4/V5 loader keeps clips with 0.6 s < duration < 54 s (data_utils.py:588).
- The sampler buckets are [32 … 1000] frames (s2_train_v3_lora.py:108-122). One frame is 20 ms (hop 640 at 32 kHz, configs/s2.json), and clips outside the buckets are discarded (data_utils.py:979-983). In practice the LoRA trains on 0.64–20 s.
- Clips are de-duplicated by file name: `list(set(phoneme_data) & names4 & names5)` (data_utils.py:544).
- The loader only repeats data automatically when there are fewer than 100 clips (data_utils.py:547-551). The GPT dataset does the same (dataset.py:179).
B2. What the LoRA trains
- LoRA goes only on the DiT attention layers: `target_modules=["to_k","to_q","to_v","to_out.0"]`, with `r = lora_alpha = rank` (s2_train_v3_lora.py:154-159).
- `freeze_quantizer: true` (s2.json) freezes ssl_proj, quantizer and enc_p (models.py:1310-1314 @3b95794c).
- The other non-cfm parts stay trainable and get saved; by my reading of the code these are ref_enc, bridge and wns1. The optimizer takes every `requires_grad` parameter, and the save keeps every key not in `no_grad_names`.
- Official LoRA rank choices are 16/32/64/128, default 32 (package webui.py:1736-1740).
B3. The diffusion (CFM) loss skips the start of each clip
- Each clip is split into a random prompt of 0 to 2/3 of its length (models.py:1338-1339 @3b95794c). The loss counts only the frames after the prompt (models_v5.py:31-47).
- Derived from the code: a frame at fraction p of the clip is in the loss with probability min(1, 1.5p). Sentence onsets (the first ~third) are therefore trained less often than endings.
B4. GPT training
- Clips are kept only if they have 3–25 phones per second (dataset.py:170) and are at most 54 s long (`max_sec`, s1longer-v2.yaml; dataset.py:146).
- There is no validation at all (s1_train.py:116 `limit_val_batches=0`).
- The learning rate is hard-locked at 0.002: `self.lr = lr = self.end_lr = 0.002  ###…直接锁定！` (AR/modules/lr_schedulers.py:58 @3b95794c). The warmup and decay settings in the yaml have no effect, so every extra epoch is an equal-sized step.
- The official GPT default is 15 epochs, max 50, DPO off and labelled experimental (webui.py:1799-1820).
B5. The SoVITS LoRA has no evaluation either
- The trainer passes `[train_loader, None]` (s2_train_v3_lora.py:249,262).
- Official defaults for v3/v4/v5: 2 epochs, max 16, with the comment `#训太多=作死` (webui.py:129-131); batch = VRAM GB // 8 (webui.py:119).
- The official wiki line (recorded in research/gsv_v4_v5/记录.md:15) is "基于 v5 版本的少量样本微调暂未进行研究和实验": nobody official has published V5 fine-tuning results.
B6. The V5 synthesis window
- The mel runs at 100 frames/s (mel_fn_v4: hop 320 at 32 kHz, TTS.py:159-166).
- `V5_REFERENCE_FRAMES=500`: only the last 5 s of the reference audio is used as the prompt (models_v5.py:11, 56).
- Targets are generated in chunks of min(1000 − ref, 640) frames, so 5–6.4 s per chunk and at most 10 s of context in total (models_v5.py:12, 59).
- Training clips of about 3–12 s therefore already cover the context that synthesis uses.
B7. Official text handling of mixed lines
- Official 1-get-text runs `clean_text(text, "zh")` on the whole line (1-get-text.py:92). chinese2 then deletes everything that is not a Chinese character or punctuation (chinese2.py:68) and also removes `[a-zA-Z]+` (chinese2.py:185). English inside Chinese lines never reaches training.
- At synthesis, TextPreprocessor (3b95794c, size 9235 = package):
  - it splits the text with LangSegmenter and merges neighbouring segments of the same kind, English with English and non-English with non-English (lines 157-169);
  - English segments go to ARPAbet with zero BERT features (lines 212-220).
- VoiceTwin's gsv_mixed_text.py and gsv_scripts/get_text_mixed.py mirror this synthesis logic. Its docstring cites commit abe9843. I compared the merge logic by eye with 3b95794c and it matches; a one-line regression test against 3b95794c would make that certain.
- cleaner.py:48-49 puts a "," pause symbol in front of any English segment with fewer than 4 phones. This is the same in training and synthesis, so it is consistent. Do not "fix" it on one side only.
B8. The official feature step re-normalises every clip
- 2-get-hubert-wav32k.py:60-61,87 writes each clip as 50% peak-normalised-to-0.95 plus 50% original.
- So whatever per-file loudness VoiceTwin sets is partly overridden per clip anyway.

=== C. Her material, measured ===
Source: voicetwin/data/lexicon/core_corpus.tsv. Analysis script: /tmp/claude-0/-home-user/3d87cc6c-41dd-58c2-802f-d51dc0c78527/scratchpad/analyze_corpus.py, with the official cmudict.rep and cmudict-fast.rep.
- 1005 lines (the docs say 1004): 559 mixed, 446 pure Chinese, 0 pure English.
- 2080 English tokens, 340 distinct.
- Only 4 English tokens are missing from CMUdict:
  - `vr`, `ppt`: 3 letters or fewer, so they are read as letters (english.py qryword), which is fine.
  - `LvMeng` (line 273, the pinyin of 吕蒙): would go through the neural g2p guess.
  - `JimWho` (line 638, "Jim … who" run together): same problem.
  These two transcripts should be fixed to what she actually said.
- `AI` is removed from CMU on purpose by en_G2p, so it is read as letters, which is fine.
- English spans of 6 or more words inside Chinese lines: 90. Of 10 or more words: 21. So her English sentence prosody does exist in the data, embedded in Chinese lines.
- 53 lines contain a question mark; 1 contains an exclamation mark.
- Line length: median 41 units (Chinese characters + English words), p90 62, minimum 8.
- 嗯/呃/啊 occur 0 times; 那么 74, 就是说 69, 然后 59.
- Whisper is known to drop fillers: the CrisperWhisper paper says "Whisper eliminates many filler words, recurring utterances" (abstract via search, https://arxiv.org/abs/2408.16589). Either she reads from a script and has no fillers, or her fillers are in the audio but missing from the text. Listen to 20 random clips to find out (cost: minutes). If fillers are missing from the text, the GPT learns them as uncontrolled insertions.

=== D. VoiceTwin prepare: what can drop or damage good material (ranked) ===
The numbers to check on her PC are manifest `drop_reason` counts, split by group (mixed / pure Chinese / new session).

D1. ASR-confidence drops on code-switched lines
- Where: prepare.py:494-504, default_config.yaml:38-39 (avg_logprob < −1.0, no_speech_prob > 0.6).
- Only lines whose text was edited are exempt (line 499). A line that matches the 母本 exactly, or that the teacher confirmed without editing, is still dropped when Whisper had low confidence.
- Whisper is weaker on Mandarin-English code-switching. This comes from code-switch papers found by search, e.g. https://arxiv.org/pdf/2305.11095; I did not verify their numbers.
- Risk: the mixed group (55%, the most important one) loses more lines than the others.
- Fix: exempt any line whose text equals the 母本 or was confirmed, from the confidence filters.
- Verify: drop counts for mixed lines vs pure-Chinese lines.
D2. The voiceprint filter against the new 26-min session
- Where: prepare.py:528-566. Threshold = clip(median − 3.5·MAD, 0.55, 0.8) (line 556); drop at line 565.
- About 2 h of old recordings dominate the voiceprint average. Speaker-verification embeddings are channel-sensitive (JUDGEMENT), so a new mic or room can push the new clips below the threshold. Expressive lines are at risk too.
- Fix:
  - log similarity per source;
  - compute the average per source;
  - only flag, never drop, when a source is a single-speaker studio file.
D3. Short isolated utterances
- min_duration 2.0 (default_config.yaml:27) drops "太短" clips (prepare.py:487-488). Lines with fewer than 2 syllables are dropped as "没有识别出文字" (482). Clips under 0.5 s are dropped silently (390).
- slicer.py:117-141 only merges a short region when the gap is ≤ 1.2 s and the merged clip stays ≤ 12 s. A short "好。/对。" between two long sentences, or after a long pause, is lost.
- The official code would accept these: the LoRA takes > 0.64 s and the GPT has no minimum (B1, B4).
- Fix:
  - train on clips ≥ 1.0 s;
  - allow 1-syllable lines when the text is confirmed;
  - keep the 3–9 s rule only for validation clips and references.
D4. The hallucination list contains real Bilibili phrases
- asr.py:247-266 lists 点赞/转发/订阅/打赏. Any line under 40 characters containing one of them becomes "疑似识别错误".
- This check runs before the human-edit exemption (prepare.py:483-484), so even a corrected "记得点赞转发" is dropped.
- Fix: skip the check for edited or 母本-confirmed text.
D5. Automatic denoising on gated recordings
- Where: enhance.py:209-214; SNR = p95 − p10 of non-zero frames (enhance.py:175-181).
- Digital-zero gaps are excluded from the SNR. If her pauses are true digital silence (she says 句子之间完全静音), p10 falls on soft speech, so the estimated SNR can drop below 25 dB. Then noisereduce would use her own soft speech and breaths as the "noise" profile.
- Fix:
  - set `denoise: off` for her studio material;
  - check `sources.json` (`snr_before`, `denoised`) on her PC.
D6. Forced cuts in the middle of phrases
- _split_long (slicer.py:49-67) cuts at energy minima, and forced pieces get 0 padding (slicer.py:156-157).
- The ASR then transcribes half-words at the cut, so the text no longer matches the audio, which hurts the GPT.
- Fix:
  - count `forced_cuts > 0`;
  - when a region has no pause of 300 ms or more, allow up to about 15 s (the LoRA accepts up to 20 s, the GPT up to 54 s) instead of cutting;
  - or keep forced pieces out of GPT training only.
D7. Speech-rate outliers
- prepare.py:506-522: < 0.5× or > 1.8× the median rate.
- Deliberate slow emphasis or drawn-out explanations can be dropped. Low risk; count them.
D8. Subtitle (SRT) path
- refine_boundaries only searches ±0.3 s (subtitles.py:135-151); onsets earlier than that get cut.
- group_cues has no 1.2 s long-pause guard (101-132).
- A single cue longer than 13 s becomes "太长".
- Subtitle text often omits fillers and repeats.
D9. Fallback text frontend and language label
- If the mixed frontend fails, VoiceTwin falls back to the official 1A (gptsovits.py:2676-2692; message at 2682). English in Chinese lines is then deleted (B7).
- exporters.py:33 writes the `detect_lang` label. A line labelled "en" that contains Chinese characters loses its Chinese in the fallback (get_text_mixed.py:153 handles this, the official path does not).
- Fixes:
  - write `send_lang(text, lang)` as the label;
  - in 一模一样 mode, treat the fallback as a quality stop (it is one), not a warning.
D10. The validation holdout (20 clips, about 2%; prepare.py:578-606) is fine; keep it.
D11. Unsourced claim shown to the teacher
- prepare.py:851 tells her "1~3 小时效果最好". I found no source for this. Rule 7 says to soften it or drop it.

=== E. How much data and what clip lengths ===
- There is no official number for V5. Official guidance is "1 min" few-shot (README/wiki), plus "V5 fine-tuning not researched" (B5).
- She has about 2.4 h, so quantity is not the bottleneck; transcript accuracy, what the filters drop, and English coverage are.
- Clip length (JUDGEMENT backed by B1/B3/B6):
  - target whole sentences of 3–10 s;
  - accept 1–15 s for training;
  - avoid long internal silences (the GPT ps-ratio floor of 3/s drops clips with more silence than speech).
- Keep breaths inside clips. Templates trained with breaths and fillers reproduce them, in placement and type similar to the corpus (Székely et al. SSW 2019 "How to train your fillers", ICASSP 2020 "Breathing and speech planning", abstracts via search).
- Whether breaths should survive into the output is a product decision tied to the absolute-silence rule (edge trimming in the engine).
- Optional UNTESTED experiment: train on overlapping two-sentence clips (sentence k−1 + pause + k). Sentence k's onset then lands in the loss region (B3), which mirrors synthesis (prompt from another sentence, then the target). Run it as an A/B test only.

=== F. Transcript accuracy ===
- Evidence: seq2seq TTS is "only partly robust" to transcript errors; substitutions and deletions hurt, extra words are partly tolerated (Fong et al., Interspeech 2019, https://www.isca-archive.org/interspeech_2019/fong19_interspeech.html, abstract).
- For the GPT, every error is a wrong phone-to-prosody pair.
- Actions:
  1. Fix LvMeng/JimWho.
  2. Run an automatic OOV list against CMUdict for every training line.
  3. Run a polyphone audit: G2PW pinyin compared with the pinyin of a second ASR engine. The package bundles Paraformer-large zh offline; see 文件清单 tools/asr/models.
  4. Add her personal readings to the official override files: `GPT_SoVITS/text/engdict-hot.rep` (English word → ARPAbet, applied after the cache, english.py `hot_reload_hot`) and `text/g2pw/polyphonic.rep` (rebuilt when its md5 changes, chinese2.py:15-18,128-134). Both affect training and synthesis alike. They live inside the package, so reinstalling the package loses them.
  5. Do the filler check from section C.
  6. Use cross-checking with a second ASR engine on the final text, not on raw ASR output.

=== G. Loudness and denoising ===
- Keep −20 LUFS per source file with whole-file peak scaling (enhance.py:224; audio.py:419-424). It preserves dynamics, and the official per-clip normalisation follows anyway (B8). No limiter or compressor.
- Denoising: the official changelog says its denoiser outputs 16 kHz "除非底噪很大否则不急使用" (Changelog_CN.md:218).
- Studies (abstracts/snippets): denoising helps similarity on noisy data but degrades it otherwise (https://arxiv.org/pdf/2210.06370). Enhancement removed some voice information and made adaptation unstable (https://arxiv.org/pdf/2005.12531).
- Conclusion: `denoise: off` for her (D5). Keep `separate_vocals: false`.

=== H. Mixed Chinese-English, and making English sound like her ===
- What V5 does with English inside Chinese: v2 symbols; English as ARPAbet through CMU, the hot dictionary, or neural g2p; BERT features zero for English; Chinese through chinese2 + G2PW (needs opencc). This is already mirrored in training (B7).
- Making English hers, in order:
  - (a) Keep all 559 mixed lines; fix D1/D9.
  - (b) Record 10–20 min of her reading typical English example sentences the way she does in class. Full English sentences at synthesis are otherwise outside her fine-tuning distribution. JUDGEMENT; VoiceTwin already warns about this at prepare.py:853.
  - (c) Use hot-dictionary entries for her idiosyncratic pronunciations.
  - (d) For English-heavy target sentences, choose a reference whose last 5 s contain English (B6). UNTESTED.
- Fun-ASR-Nano (official integration 2026-08-18) self-reports lower Chinese WER than Whisper-large-v3: AIShell1 1.80 vs 4.72, Fleurs-zh 2.56 vs 5.18 (https://github.com/FunAudioLLM/Fun-ASR). It reports no code-switching test, so it is a candidate second engine for verification only.

=== I. Duplicating or weighting rare data ===
- The official loaders never oversample above 100 clips, and de-duplicate by file name (B1). Weighting therefore needs renamed copies (e.g. `x__dup1.wav`, with its .pt features copied), never the same name twice.
- Evidence that balancing minority languages matters in multilingual TTS: https://www.arxiv.org/pdf/2110.04482v1 (abstract via search); Coqui's `use_language_weighted_sampler`. Nothing specific to GPT-SoVITS.
- JUDGEMENT: at most ×2. Candidates are the 90 long-English-span lines, the 53 question lines, and the new session if that is the target sound. Never duplicate validation lines. Keep it only if the per-group held-out score improves.

=== J. Detecting under- and over-fitting; how many epochs ===
- Evidence:
  - official: V5 LoRA default 2 epochs, max 16 "训太多=作死"; GPT 15 epochs at a constant learning rate (B4, B5);
  - one community v4 LoRA case: about 467 Japanese clips, rank 32, batch 4. At 8 epochs SV similarity was 86.9% of the real-voice baseline; at 16 epochs it was 91.5% (its GPT epochs also changed). The 16-epoch model had an "electronic tone", and CER missed emotion and pronunciation errors (https://github.com/Star-Meow/Ryza_voice/pull/6);
  - community guide: too many GPT epochs leads to skipped or repeated words; too many SoVITS epochs leads to electrical noise, high-frequency artifacts and learned noise floor (https://www.juyaohuang.com/blog/ai/gptsovits/gpt-sovits-model-choose).
- VoiceTwin today: LoRA 4 (standard) / 8 (一模一样) epochs, saving every epoch (gptsovits.py:1011-1013); GPT 15/20 epochs.
- With about 1000 clips at batch 2 (16 GB), one epoch is about 500 steps, so 8 epochs is about 8000 sample passes. That is similar in exposure to the community 16×467 run (arithmetic, not a measurement).
- Recommendations:
  - (1) Per saved checkpoint, compute held-out losses on the 20 validation clips with fixed seeds:
    - GPT: teacher-forced NLL and top-3 accuracy on her real semantic tokens. This directly measures "learned her prosody";
    - CFM: flow-matching loss with fixed t and noise.
    Both take seconds per checkpoint and need no synthesis (UNTESTED estimate). A U-shaped curve means over-fitting; still falling means under-fitting.
  - (2) Use these curves to pre-select which checkpoints get the expensive synthesis scoring.
  - (3) If the last epoch wins, continue training automatically (LoRA to 12, then 16; GPT to 25+). The official script resumes from the latest `G_*.pth` (s2_train_v3_lora.py resume block) and VoiceTwin already has 接着练.
  - (4) Add prosody metrics, because the community case shows CER and SV miss emotion: F0 contour correlation after DTW, pause and rate per group, question-final rise.
  - (5) Test LoRA rank 32 vs 64 as an A/B on her PC in 一模一样 mode. There is no evidence either way.
- DPO only reduces repeated or skipped words (Changelog_CN.md:181,193,731); it is not a similarity lever.

=== K. Other tools that could assist ===
- A second verification ASR: the bundled Paraformer-large zh, or Fun-ASR-Nano.
- Loss-based transcript cleaning: rank training clips by GPT NLL under the fine-tuned model and listen to the top outliers. JUDGEMENT ("small-loss" idea; untested here).
- engdict-hot / polyphonic overrides (F4).
- A forced aligner (e.g. MFA Mandarin) to find text/audio mismatches. UNTESTED for code-switching.
- No denoiser and no vocal separation for her studio material.

=== L. How to verify (GitHub Actions has CPU only) ===
- On CI (CPU):
  - the corpus/OOV/polyphone audit;
  - prepare-filter counts on synthetic and fixture data;
  - a frontend parity test (VoiceTwin mixed text vs TextPreprocessor @3b95794c on all 1005 lines: identical phone sequences);
  - the held-out-loss script on tiny checkpoints (the gsv-v5-real workflow already trains 2 epochs on CPU).
- On her PC:
  - manifest `drop_reason` histogram per group and source;
  - `sources.json` denoise flags;
  - training time and VRAM per epoch from the logs (not measured anywhere yet);
  - loss curves plus VoiceTwin's selection scores per epoch;
  - LoRA rank and epoch A/B tests.
- Every change above affects material only; none touches the inter-sentence digital silence.

=== M. Outside my area but relevant ===
- V5 uses only the last 5 s of the first reference (models_v5.py:11,56). Choose references so that their final 5 s are representative, or use references of 5 s or less.

=== Sources ===
- Official GPT-SoVITS code @3b95794c / main, via raw.githubusercontent.com/RVC-Boss/GPT-SoVITS:
  - GPT_SoVITS/module/data_utils.py, s2_train_v3_lora.py, module/models.py, module/models_v5.py;
  - AR/data/dataset.py, s1_train.py, AR/modules/lr_schedulers.py;
  - text/chinese2.py, cleaner.py, english.py;
  - prepare_datasets/1-get-text.py, 2-get-hubert-wav32k.py;
  - TTS_infer_pack/TextPreprocessor.py, TTS.py;
  - configs/s2.json, s1longer-v2.yaml;
  - tools/slice_audio.py;
  - docs/cn/Changelog_CN.md, docs/cn/README.md.
- Package files: research/gsv_v4_v5/官方资料/v5整合包/webui.py, 文件清单.tsv. Wiki quotes: research/gsv_v4_v5/记录.md:12-15.
- Community and papers:
  - https://github.com/Star-Meow/Ryza_voice/pull/6
  - https://www.juyaohuang.com/blog/ai/gptsovits/gpt-sovits-model-choose
  - https://www.isca-archive.org/interspeech_2019/fong19_interspeech.html
  - https://arxiv.org/abs/2408.16589
  - https://www.isca-archive.org/ssw_2019/szekely19_ssw.html
  - https://www.speech.kth.se/tts-demos/szekely2020breathing.pdf
  - https://arxiv.org/pdf/2210.06370
  - https://arxiv.org/pdf/2005.12531
  - https://www.arxiv.org/pdf/2110.04482v1
  - https://arxiv.org/pdf/2305.11095
  - https://github.com/FunAudioLLM/Fun-ASR
