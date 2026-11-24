# 调研（美化）：english-pron（2026-11-24）

RESEARCH REPORT: polishing English without changing the teacher's voice (VoiceTwin / GPT-SoVITS V5). Research only; no project files were edited.

## 0. Bottom line (in this order)
1. **Fix the text side first (most of the gain, no identity risk).** I ran V5's own English front end on the teacher's own lecture text. It already produces wrong phones in her real sentences: "AND" is read as letters A-N-D; "Dr.Banner" becomes "drive . Banner"; "lead to" uses the metal reading; the citation word "and" gets the weak schwa form. These are deterministic bugs. Fix them with:
   - a teacher lexicon,
   - a program-run "英文读音预检" (English pronunciation pre-check) before synthesis,
   - small text rewrites.
   Apply the fixes identically in training (gsv_scripts/get_text_mixed.py) and inference (api_v2), so labels and generation stay consistent.
2. **Optional "美化" (beautified) version through selection only.** Add an open-source phoneme-level pronunciation score (CTC phoneme recognizer plus GOP). Use it to:
   - pick, among candidates that already pass every existing gate, the one with the best English;
   - rank the teacher's own English reference clips. V5 uses only the first reference (handbook §16.6), so this choice matters.
   No audio is altered, so this is identity-preserving by construction.
3. **Do not ship accent conversion now.** No permissively licensed, GPT-SoVITS-compatible, identity-safe system exists. The research systems trade accent against speaker identity and target a "native" (usually American) accent.

## 1. Verified facts about V5's English front end (I ran the code)
- **The V5 package front end matches upstream main by file size.** Sizes come from research/gsv_v4_v5/官方资料/v5整合包/文件清单.tsv, compared with raw.githubusercontent.com RVC-Boss/GPT-SoVITS main:
  - text/english.py 9803
  - cmudict.rep 3731285
  - cmudict-fast.rep 3613898
  - engdict-hot.rep 75
  - en_normalization/expend.py 10879
  I ran the upstream files with g2p_en 2.1.0 (Apache, from PyPI), the same version as V5's runtime_包版本.txt.
- **Lookup order in english.py `en_G2p.__call__`:**
  1. single letters;
  2. homographs (g2p_en's 379-line homographs.en, decided by NLTK part-of-speech tag);
  3. `qryword`: CMU dict, which engdict-hot.rep overrides at load (`hot_reload_hot`);
  4. out-of-dictionary words of 3 letters or fewer are spelled out;
  5. wordsegment split;
  6. g2p_en neural guess.
  Consequences:
  - **Homographs ignore the hot dictionary.**
  - **The CMU dict keeps only the first pronunciation.** `read_dict_new` stores "live(1)" as its own key, so plain "live" always gets the first entry.
- **English gets zero BERT features.** `TextPreprocessor.get_bert_inf` builds BERT only for zh. So English phrasing and prosody come only from the phones, the punctuation and the reference audio.
- **cleaner.py adds a pause token before short English segments.** It prepends "," when an English segment has fewer than 4 phones. This hits most of her single-word mentions (and, which, that, whom, who, as, whose).
- **Probe results.** Scripts are in the scratchpad (paths in section 7). All outputs below are from running them:
  - "This will lead to a crisis." → L EH1 D (wrong; should be L IY1 D). "I live in Beijing." → L AY1 V (wrong). "The wind…" → W AY1 N D (wrong). "a tear in her eye" → T EH1 R (wrong).
  - "object": the g2p_en entry is swapped. "I object…" → AA1 B JH EH0 K T and "an object" → AH0 B JH EH1 K T; both are wrong.
  - "Please read it." → R EH1 D (wrong). A bare "read" → R EH1 D.
  - All-caps words are always spelled letter by letter (expend.py line 316 puts a space before each capital). COVID → "S IY1 OW1 V IY1 AY1 D IY1"; NASA, UNESCO and AND are spelled out too. "Nasa" → N AE1 S AH0 is correct, but "Covid" is still wrong (K OW1 V IY1 AY1 D IY1).
  - "Dr"/"Dr." → D R AY1 V ("drive"). In "Mr.", "U.S." and "No." the period becomes a "." pause token.
  - "1990s" → "nineteen ninety seconds". "postgraduate" → P AH0 S T AE1 G R AH0 D EY2 T (wrong). "metaverse" → M AH0 T AA1 V ER0 S (wrong).
  - ";" and ":" inside English are deleted, so there is no phrase break. The cause: rep_map's key "[;:：，；]" is passed through re.escape, so it only matches that literal string, and normalize() then strips the characters. Example: "However, the data show otherwise; we must act." → "…otherwise we must act".
- **Audit of the teacher's own published corpus** (voicetwin/data/lexicon/core_corpus.tsv, 1005 sentences):
  - 559 sentences contain English: 2080 word tokens, 379 distinct.
  - 974 English chunks, of which 647 are single-word mentions (top: and ×73, which ×65, that ×55, whom ×39, who ×36, as ×30, whose ×26).
  - Real errors found: "AND" ×10 in all caps ("AND的第二种连接方式") → EY1 EH1 N D IY1 (letters). "Dr.Banner" → "D R AY1 V . B AE1 N ER0". "LvMeng" → spelled "L V" + MEHNG.
  - Citation forms of isolated words: and → AH0 N D, the → DH AH0, a → AH0 (weak forms).
  - Not verified: whether the trained model actually voices AH0 N D as her stressed /ænd/. Training used the same g2p labels, so it might. This needs listening and the scorer.
- **Runtime patching works without editing package files.** Adding entries to `text.english._g2p.cmu` at runtime changed the output: "leadverb to" → L IY1 D T UW1, "windnoun" → W IH1 N D, "postgraduate" → P OW2 S T G R AE1 JH UW0 AH0 T. Pseudo-words must be lowercase letters only (capitals are split apart), more than 3 letters, and not in homographs.en.

## 2. Text-side fixes (the task's item 2): implementable design
- **Pre-check (`英文读音预检`).** Run the package's own english.py with the V5 runtime Python on every English chunk of the script. List for the teacher:
  - (a) all-caps words that will be spelled out;
  - (b) abbreviations with periods: Dr., Mr., U.S., No., e.g.;
  - (c) digits followed by "s";
  - (d) ";" and ":" inside English;
  - (e) words outside the dictionary that get a neural guess;
  - (f) CMU multi-entry words where the first entry is taken: lead, live, wind, tear, minute…;
  - (g) g2p_en homographs, whose result depends on part-of-speech tagging and which have known-bad entries (object, read);
  - (h) isolated function-word mentions (strong vs weak form).
  Each item shows the planned phones and a one-click choice. This is deterministic and program-run, which fits the project's "机器执行、坏样本必抓" rule.
- **Teacher lexicon.** Store it in the voice folder as word → ARPAbet with stress. GPT-SoVITS symbols2 contains stress-marked ARPAbet (AH0/AH1/AH2…), so lexical stress is controllable. Per-occurrence overrides use simple markup that is turned into pseudo-words. Apply it by runtime patch in:
  - get_text_mixed.py (training);
  - a small launcher that patches `_g2p.cmu` and then runs api_v2.py through runpy (inference).
  The alternative is writing GPT_SoVITS/text/engdict-hot.rep. It also covers both paths, but edits the package and is lost on reinstall.
  Required alongside:
  - put the lexicon hash in the synthesis cache key (engine.py `parts`);
  - restart the server after a lexicon change;
  - retrain when training labels change. The labels should match what she actually said, e.g. /liːd/ in "lead to".
- **Text rewrites in the engine copy only.** Subtitles and the CER reference keep the original text:
  - ;/: → ",";
  - Dr. → Doctor;
  - 1990s → nineteen-nineties;
  - acronyms read as words → lowercase, plus a lexicon entry where still out of the dictionary (covid);
  - add commas at clause boundaries in long English sentences to get her phrasing (punctuation tokens: ! ? … , . -).
- **Identity and verification.** No audio is processed, so timbre is untouched. Gates:
  - 像你本人 % within measurement noise of the current output;
  - English WER not higher;
  - bad samples must be flagged: AND, Dr.Banner, "lead to", 1990s, "I object";
  - good samples must pass.
  Effect on audio is not yet measured.
- **Digital silence.** Unaffected: the change is upstream of per-sentence generation, and assembly still inserts exact-zero gaps.

## 3. Automatic pronunciation scoring for candidate selection (the task's item 1)
**Published results (speechocean762 = 5000 utterances, Mandarin-L1 speakers, 5 expert raters; licence CC BY 4.0, "free … commercial and non-commercial").** Rows are ordered by confidence in the source, best first:

| Source | What it reports | Licence / access | Confidence |
|---|---|---|---|
| Kaldi GOP baseline (egs/gop_speechocean762) | phone-level GOP-NN | Kaldi Apache-2.0 | Repo checked |
| GOPT (Gong et al., ICASSP 2022), README | Librispeech model: phone PCC 0.616, word total 0.552, utterance total 0.743. Paper headline: 0.612 / 0.549 / 0.742 | BSD-3 | README checked. Needs Kaldi GOP features (librispeech m13): impractical on Windows |
| HiPAMA (arXiv 2211.08102v1 table) | phone 0.616, word accuracy 0.575, utterance total 0.754 | — | Search snippet only |
| Segmentation-free GOP (Cao, Fan, Svendsen, Salvi; IEEE TASLP; code github.com/frank613/CTC-based-GOP) | 0.648 PCC with a GOPT-style model | Repo has **no LICENSE file** (404 checked): reimplement from the paper, don't copy | 0.648 is from a search summary, unverified |
| Parikh et al. (Interspeech 2025), logit-based GOP | Logit methods beat probability GOP for mispronunciation classification; advantage depends on dataset; margin score generalised poorly on speechocean762 | Uses facebook/wav2vec2-xlsr-53-espeak-cv-ft (Apache-2.0 per HF listing; arXiv 2109.11680; outputs espeak IPA phones) | Search summary |
| Discrete-token surprisal (arXiv 2606.19910) | PCC 0.60 → 0.66 | — | Search summary |
| Microsoft blog | PCC 0.5661 → 0.6562; GOPT baseline 0.5950; estimated "human parity" 0.6787 | Commercial | Self-reported |
| Fine-tuned large multimodal models (arXiv 2509.15701) | PCC about 0.9 but rank correlation (SCC) about 0.6 | — | Search snippet |

- **The last row is the key caveat for selection.** A high correlation across speakers does not mean reliable ranking of near-identical candidates. Ranking reliability must be measured on our own candidates.
- **Why the existing Whisper check is not enough.** A word-level ASR with a language model auto-corrects substitutions and distortions (Academia Sinica child-speech paper; NLP4CALL 2024 Whisper study). A 2026 NLP4CALL paper found that Whisper low-probability scores track mispronunciations only uncertainly.

**Recommended scorer (most current method that is practical here):**
- **Model.** CTC phoneme recognizer wav2vec2-xlsr-53-espeak-cv-ft. The V5 runtime already has transformers 4.51.3 and torch 2.7.1+cu128, so it runs on RTX 50-series or CPU.
- **Scores.** Reimplemented segmentation-free CTC GOP per phone, against canonical phones taken from the corrected lexicon (not raw g2p output, otherwise g2p errors are invisible). ARPAbet is mapped to espeak IPA with a fixed table. Then derive:
  - word-level minimum GOP (catches one bad word);
  - completeness: deletions found by the CTC alignment;
  - phone error rate.
- **English spans inside Chinese sentences.** Cut them using the existing faster-whisper word timestamps. Unverified design; test it.
- **Use for the original version.** Ranking unchanged, plus a hard filter on gross English errors.
- **Use for the 美化 (beautified) version.** Among candidates that already pass every gate, take the best English GOP, subject to:
  - 像你本人 ≥ (best − δ), where δ is the measured noise of 像你本人 (e.g. its spread when re-scoring her held-out real recordings; not yet measured);
  - English WER no higher than the original version's;
  - speaking rate and pitch inside her range.
- **References.** Rank her English-containing reference clips by the same GOP and prefer high scorers for English-heavy sentences. These are her own recordings, so identity is safe. Training clips are not dropped (handbook rule 15: every clip must be trained on).
- **Calibration.** Like 像你本人: 100% = her real English recordings.
- **Automatic validation:**
  1. reproduce a published-range phone PCC on the speechocean762 test set (2500 utterances);
  2. bad samples rank last: deliberately wrong lexicon (L EH1 D in "lead to", spelled AND, skipped word);
  3. measure within-sentence ranking with the existing ⑤ blind test on about 20 pairs.
- **Cost.** Score only English-containing sentences and only the top-K candidates. Speed is not measured.
- **Digital silence.** Scoring is analysis only and runs per sentence on the exact trimmed clip before assembly. The existing post-assembly gate (gaps exactly 0) is unchanged.

## 4. Accent or pronunciation conversion (the task's item 3): not safe or ready
- **Vevo (Amphion).** Zero-shot accent conversion, but the weights are cc-by-nc-4.0 per the HF listing (seen via search). Non-commercial conflicts with monetised Bilibili videos.
- **FAC-FACodec (arXiv 2510.10785).** Has an explicit accent-strength vs identity trade-off and pushes toward a standard American accent. No code found.
- **Activation steering (arXiv 2603.05977, Interspeech 2026 submission).** Built on Qwen3-TTS (a listed backup engine). Reports timbre kept "to a large extent". Code not confirmed.
- **Accent Analogy Guidance and closed-loop guidance (arXiv 2609.29123, 2610.29351).** Research on cross-lingual cloning, not GPT-SoVITS.
- **Common risk.** All of these re-synthesise or steer the whole utterance. Accent is part of what speaker embeddings capture, so 像你本人 would likely drop (unmeasured).
- **If ever trialled.** Only as a third optional track, per sentence before assembly. Gates: 像你本人 drop ≤ δ, WER not up, GOP up, blind test passed; then re-check edges and digital silence.

## 5. Unverified / open
- Audible effect of the lexicon fixes on the trained V5 model.
- δ (the noise of 像你本人).
- How well GOP ranks candidates within one sentence.
- Whether the espeak-to-ARPAbet mapping is good enough.
- Licences of the lv-60 espeak model and of community speechocean762 models (e.g. WavLM "accent-scorerer", self-reported 0.7487): unverified, not recommended.
- arxiv.org, huggingface.co and accentsteer.github.io were blocked in this session. Paper numbers marked "search" came from search summaries and should be confirmed from the PDFs.

## 6. Sources
GPT-SoVITS main, raw.githubusercontent.com/RVC-Boss/GPT-SoVITS:
- GPT_SoVITS/text/english.py
- engdict-hot.rep
- cleaner.py
- symbols.py, symbols2.py
- en_normalization/expend.py
- TTS_infer_pack/TextPreprocessor.py

Other code and data:
- g2p_en 2.1.0 wheel (pypi.org) and its homographs.en
- github.com/YuanGongND/gopt (README, LICENSE)
- github.com/jimbozhang/speechocean762 (README)
- kaldi-asr/kaldi egs/gop_speechocean762/README.md and COPYING
- github.com/frank613/CTC-based-GOP (README; no LICENSE)
- huggingface.co/facebook/wav2vec2-xlsr-53-espeak-cv-ft (via search)
- huggingface.co/amphion/Vevo (via search)

Papers and posts:
- arXiv 2205.03432 (GOPT)
- arXiv 2211.08102 (HiPAMA)
- arXiv 2507.16838 (Segmentation-free GOP)
- arXiv 2506.12067 (Parikh et al., Interspeech 2025)
- arXiv 2606.19910 (discrete-token surprisal)
- arXiv 2503.11229 and 2509.15701 (large multimodal models for assessment)
- techcommunity.microsoft.com pronunciation-assessment posts
- aclanthology 2024.nlp4call-1.2 and 2026.nlp4call-1.8; ling.sinica.edu.tw child-speech paper
- arXiv 2510.10785 (FAC-FACodec), 2603.05977 (activation steering), 2609.29123 (Accent Analogy Guidance), 2610.29351 (closed-loop dual-reference guidance)

Repo files:
- /home/user/voicetwin/voicetwin/backends/gsv_mixed_text.py
- /home/user/voicetwin/voicetwin/backends/gptsovits.py (api_v2 launch at lines 3790–3792)
- /home/user/voicetwin/voicetwin/synth/engine.py (cache key at lines 866–877)
- /home/user/voicetwin/教学手册.md §16.6

## 7. Probe scripts (scratchpad, re-runnable)
Folder: /tmp/claude-0/-home-user/3d87cc6c-41dd-58c2-802f-d51dc0c78527/scratchpad/scripts/
- g2p_probe.py
- g2p_probe2.py
- corpus_audit.py
- corpus_ctx.py
- iso.py

Run with:
`NLTK_DATA=/tmp/claude-0/-home-user/3d87cc6c-41dd-58c2-802f-d51dc0c78527/scratchpad/nltk_data /tmp/claude-0/-home-user/3d87cc6c-41dd-58c2-802f-d51dc0c78527/scratchpad/venv/bin/python -I <script> /tmp/claude-0/-home-user/3d87cc6c-41dd-58c2-802f-d51dc0c78527/scratchpad/gsv [corpus.tsv]`
