# 调研（美化）：mastering（2026-11-24）

## Research report: classic voice mastering and a "light beautify" option for VoiceTwin

No project files were edited. My test scripts are in the scratchpad (`/tmp/claude-0/-home-user/3d87cc6c-41dd-58c2-802f-d51dc0c78527/scratchpad/silence_test.py`, `codec_test.py`). Their numbers come from a **synthetic** voice-like signal, not from the teacher's voice.

Several hosts are blocked in this session: bilibili.com, podcasters.apple.com, tech.ebu.ch, itu.int, productionadvice.co.uk, arxiv, github.com pages. For those, the facts below come from search-result summaries and the pages were not opened ("via search"). I read PyPI pages, raw GitHub READMEs and the installed package source/docstrings directly ("verified").

### 0. Bottom line
1. **Version A (原声, unprocessed) stays the default.** Build the optional **Version B "轻度美化"** in four steps:
   - (a) A bounded spectral correction toward *her own real recordings*, measured as the long-term average spectrum (LTAS).
   - (b) A small bounded EQ for warmth, presence and air.
   - (c) A de-esser and a gentle compressor. Their thresholds come from *her own recordings'* statistics, not from fixed presets.
   - (d) A per-sentence true-peak limiter, then the existing per-sentence gain and zero-track assembly.
2. **Every step runs per sentence, before assembly**, on a zero-padded copy that is then trimmed and faded back. Running it on the assembled track breaks absolute digital silence (measured in §3).
3. **Safety gates, all automatic:**
   - Paired 像你本人 % change and ASR error rate, processed vs unprocessed, per sentence.
   - The chain is also calibrated once on her real held-out recordings.
   - Because the speaker models and ASR only see audio below 8 kHz, there is an extra LTAS / sibilance / noise-floor guard above 8 kHz.
   - An exact-zero count, and a true-peak check.
   - A sentence that fails falls back to unprocessed audio inside Version B.
4. **Mastering cannot make English pronunciation "more polished".** A presence boost only helps perceived clarity. Pronunciation lives in the TTS model or in candidate selection.
5. **Exciter/saturation is left out of the light preset.** There is no reputable parameter guidance for speech, it is nonlinear, and the identity check cannot see the band it mainly changes (above 8 kHz).

### 1. What the app already does (verified in code)
- `engine.py::_levels` gives each sentence one fixed gain matched to her loudness distribution. It never uses a limiter. Any sentence whose **sample** peak would go above −1 dBFS (`PEAK_CEILING_DB`) is lowered on its own.
- `_render` puts each sentence into an all-zero track after a 4 ms / 8 ms fade.
- The loudness target is her source recordings' median LUFS (`_target_lufs`, profile), otherwise −18.
- `_make_variants` already builds A/B versions on the same layout and scores them with `_version_score`. In 「一模一样」 mode that call uses `use_asr=False`, so version scoring currently does **not** check ASR error.
- The speaker-verification front end (`eval/sv_frontend.py`, SR = 16000) and ASR (16 kHz) are both **blind above 8 kHz**.
- Measured noise of 像你本人 % under processing (`research/sv_benchmark/记录.md`, combined model, mean ± SD in points):

| Processing | Change in 像你本人 % |
|---|---|
| volume to max | −0.1 ± 0.9 |
| volume −14 dB | −0.6 ± 3.6 |
| denoise (version B) + absolute silence | −2.5 ± 3.1 |
| silence-only (pauses replaced by digital zeros) | −2.6 ± 4.3 |
| white noise at 20 dB SNR | −7.7 ± 7.0 |

  There is no EQ or compression entry yet, so that needs to be measured.
- The V5 runtime (`research/gsv_v4_v5/官方资料/v5整合包/runtime_包版本.txt`) has Python 3.12, numpy 1.26.4 and scipy 1.17.1. Its bundled `runtime/ffmpeg.exe` is dated 2021-02-28, so newer ffmpeg filters may be missing. Check at run time with `ffmpeg -filters`; this is unverified.

### 2. Loudness and peak targets

| Standard | Numbers | Status |
|---|---|---|
| Apple Podcasts | about −16 LKFS ±1 dB; true peak ≤ −1 dBFS; per ITU-R BS.1770-5; condition before encoding ("recommend") | via search |
| EBU R128 | −23 LUFS ±0.5 LU (broadcast); s1 short-form: max short-term −18 LUFS, true peak ≤ −1 dBTP; true-peak tolerance ±0.3 dB | via search |
| ACX audiobooks | RMS −23 to −18 dB; peak ≤ −3 dB; noise floor ≤ −60 dB RMS | via search |
| AES TD1008 (supersedes TD1004) | exact numbers not verified. A Radio World summary says −18 LUFS for speech programming | **unverified** |
| YouTube | about −14 LUFS playback normalization; turns loud videos down only; not officially documented | secondary sources, **unverified** |
| Bilibili | one person's 2024 test: mobile about −12 LUFS; web "standard" about −12, "高动态" about −20 | **unverified, unofficial** |

Recommendation:
- Keep the default "match her original recordings" target.
- Add an optional **"online video" target of −16 LUFS integrated, true peak ≤ −1.5 dBTP**. The −1.5 margin is what pedalboard's own docstring advises for its true-peak limiter.
- Reach that target with the existing gain-only leveling plus the per-sentence limiter. pyloudnorm's gating already ignores the zero gaps.
- Measure true peak with 4× oversampling. pyloudnorm 0.2.0 has no true-peak meter: its `Meter` only has `integrated_loudness` and `loudness_range` (verified). Use scipy `resample_poly(x, 4, 1)` or `ffmpeg ebur128=peak=true`.
- The current −1 dBFS check is a sample peak. True peak can be higher: in my synthetic test it was +0.18 dB higher.

### 3. Absolute digital silence: measured on the synthetic signal (48 kHz)
Test chain: HPF 70 Hz, low shelf +1.5 dB at 250 Hz, peak +1 dB at 3.5 kHz, high shelf +1.5 dB at 10 kHz, 2:1 compressor, true-peak `BrickwallLimiter` at −1.5 dB.

- **Whole assembled track through the chain:** 11,712 of 52,800 gap samples became non-zero. The tail after a sentence lasted 148 ms and reached −56.1 dBFS. This breaks the silence rule. Hard-gating afterwards would cut the tails off abruptly.
- **Per sentence (0.25 s zero padding, trim back, 4/8 ms fades, place into zeros):** 0 non-zero gap samples.
- Each block alone turns zeros into zeros, and pedalboard compensates the limiter's 5 ms lookahead (measured shift 0 samples).
- **Export:** soundfile PCM_16 and ffmpeg float→s16 kept every gap sample at exactly 0 (no dither). **MP3 and M4A export do not keep exact zeros:**
  - MP3 192k: 4,412 non-zero gap samples, up to 32 ms from speech, max −94.5 dBFS.
  - AAC 192k: 5,184 non-zero gap samples, up to 42 ms from speech, max −64.0 dBFS.

  This affects the app's existing mp3/m4a output, not just the new feature. WAV and FLAC are the only exact formats.

### 4. Tools (licences verified from PyPI metadata)
- **pedalboard 0.9.26 (Spotify), GPLv3, Windows wheels cp310–cp315.**
  - Has HighShelf/LowShelf/PeakFilter, Compressor, NoiseGate, and the new `BrickwallLimiter(ceiling_db, lookahead_ms, true_peak=True uses 4× oversampled sidechain)`. It can also host VST3 plugins.
  - Pitfalls from its docstrings:
    - `Limiter` applies automatic makeup gain that cannot be turned off, which raises quiet parts and noise. **Do not use it.**
    - `HighpassFilter` is only first-order (6 dB/oct).
    - It has no de-esser.
  - GPL note (not legal advice): the GPL does not restrict her use of the output audio. Bundling pedalboard in the MIT-licensed VoiceTwin zip brings GPL distribution duties. A GPL-free alternative is scipy biquads (RBJ cookbook) plus a numpy compressor and limiter.
  - Must be tested installed with `--no-deps` against numpy 1.26.4. Not tested here.
- **matchering 2.0.6, GPLv3.** Do **not** use it as-is. From reading its source:
  - Internal rate is 44.1 kHz; its config itself warns other rates are "not tested". She works at 48 kHz.
  - The correction ratio is unbounded (reference / target, floor 1e-6, LOWESS-smoothed).
  - It uses a 4096-tap linear-phase FIR, which pre-rings about 46 ms before every onset.
  - It is built for mid/side stereo, and its limiter targets about −0.016 dBFS sample peak.

  Its idea (FFT-ratio matching with log-domain smoothing) is worth reusing in a bounded, mono, speech-frames-only form (§5a).
- **pyloudnorm 0.2.0, MIT.** Already used for loudness.
- **ffmpeg filters:** `deesser`, `adynamicequalizer` (a dynamic EQ that works as a de-esser), `aexciter`, `alimiter`, `loudnorm`, `ebur128` (true-peak mode). Whether the 2021 bundled ffmpeg.exe has them is unverified. `loudnorm` falls back to dynamic (AGC-like) mode when linear mode is not possible, so do not use it on the assembled track.
- **timbral_models 0.4.0 (AudioCommons/Surrey), Apache-2.0.** Gives warmth, brightness, sharpness and boominess estimates. Use it only as a report value. The AudioCommons evaluation found sharpness correlated well with ratings (r ≈ 0.79, via search); how valid the warmth estimate is was not found.
- **Commercial equivalents, as benchmarks only:**
  - Auphonic (cloud): Voice AutoEQ with de-esser and de-plosive, Adaptive Leveler, loudness normalization, and a true-peak limiter with 4× oversampling (via search).
  - iZotope Ozone Match EQ: proprietary.
- **Airwindows Consolidated:** MIT for the DSP code, but the packaged binary may also carry GPL-3.0 (an AUR listing says so). Windows VST3. Possible but not needed.

### 5. Proposed design
All numbers in this section are *proposals with bounds*. None of them has been verified on her voice.

**(a) Bounded LTAS anchor toward her real voice.** This is also a candidate improvement for Version A, but adopt it there only if it measurably keeps or raises 像你本人 %.
1. Compute the LTAS (Welch, speech frames only, using the project's silero-VAD) of her held-out real recordings and of the whole generated lecture.
2. Correction C(f) = LTAS_her − LTAS_gen, level-aligned, smoothed to 1/3 octave, limited to 80 Hz–16 kHz, clamped to ±1.5 dB, tapering to 0 dB outside that range.
3. Apply it as a **minimum-phase** FIR (`scipy.signal.minimum_phase`, no pre-ringing) or as a few fitted biquads.
4. Background: LTAS is used in forensic speaker comparison (Estudios de Fonética Experimental 2023; Hughes et al., Interspeech 2017). Pulling the long-term timbre toward her own real recordings is identity-preserving by construction.

**(b) Light "magnetic / clear" preset.** One strength slider from 0 to 1 scales every gain. The total deviation from Version A must stay ≤ 2.5 dB at any frequency.

| Step | Setting (bounded) | Source / note |
|---|---|---|
| High-pass filter | 2nd-order at 60–80 Hz | Flypaper: cut below 100 Hz. Her F0 is far above this (a Taiwanese women's Mandarin sample measured about 213 Hz, SD 25, from a fragmentary source; measure hers from the profile instead) |
| Warmth ("magnetic") | low shelf at about 200–300 Hz, +0.5 to +2.0 dB, Q 0.7. Only cut 300–600 Hz build-up (−1.5 dB max) if her generated LTAS is above her real one there | Avid: warmth 300–600 Hz. Shure: close-mic proximity effect "adds warmth and fullness". No validated acoustic definition of "磁性" was found. Attractiveness studies point to F0, formants and HNR, which we deliberately do not change |
| Clarity | peak at 2.5–4 kHz, +0.5 to +1.5 dB, Q 0.8–1.2 | Avid: intelligibility 1.5–4.5 kHz. Shure: 2–10 kHz rise gives brightness and intelligibility. Bücklein (JAES 1981): peaks are far more audible than dips, so keep boosts small |
| Air | high shelf at 10–12 kHz, +0.5 to +2.0 dB, only where both her real and the generated LTAS are above the noise floor | Avid: gentle shelf from about 10 kHz |
| De-esser | dynamic, detection band 4–9 kHz, threshold = **her own p95** sibilance-band ratio, max −4 dB, attack 1–5 ms, release 50–100 ms | iZotope podcast guide: sibilance "typically 4 to 9 kHz" |
| Compressor | 1.5:1–2:1, threshold so only the loudest parts are reduced (≤ 3 dB), attack 10–30 ms, release 100–250 ms. No separate makeup gain: the existing per-sentence gain (`_levels`) restores the level | iZotope: start at 2:1, engage only on the loudest. Sound On Sound: even small compression is "very obvious" on speech |
| Limiter | `BrickwallLimiter(ceiling_db=-1.5, true_peak=True, lookahead_ms=5)` per sentence. Report how many sentences actually hit it | pedalboard docstring |
| Exciter / saturation | **off** (if ever tried: parallel, band above 7.5 kHz, gated by the above-8 kHz guard) | no speech guidance found; nonlinear; the identity check cannot see it |

**Pipeline order:**
1. Per-sentence wav, after the existing denoise choice (start from whichever of A/B is recommended).
2. Zero-pad, apply the chain, trim, fade.
3. `_levels` (gains recomputed from the processed audio).
4. `_render` into zeros.
5. Final exact-zero assertion outside sentence regions.

This reuses the `_make_variants` / `_version_score` pattern (same layout, timings and subtitles). The UI is an optional checkbox "同时做一个轻度美化版本" plus the strength slider.

### 6. Automatic verification
The thresholds are proposals, calibrated against the measured noise in §1.

1. **One-time calibration on her real held-out recordings.**
   - Run the preset on them and take the paired Δ像你本人 %.
   - Accept a strength if mean Δ ≥ −1.0 point **and** the lower 95% confidence bound ≥ −2.0. For comparison, a pure gain change gives −0.1 ± 0.9 to −0.6 ± 3.6.
   - Otherwise halve the strength and repeat.
   - This also measures how much the chain alone moves the score.
2. **Per lecture, per sentence, Version B vs Version A.** A sentence falls back to A if any of these fails:
   - Δ像你本人 % ≥ −2 × SD of the gain-only noise (about −7 points); and the sentence's % stays ≥ `min_pct` (85).
   - ASR error not higher than in A. This needs `use_asr=True` in version scoring.
   - The 1/3-octave LTAS distance to her real recordings (RMS dB, 100 Hz–16 kHz) is no more than the unprocessed distance + 1.0 dB, and no band is more than +3 dB above her own LTAS.
   - **Above-8 kHz guard** (the SV and ASR models are blind there): the 8–16 kHz to 1–4 kHz energy ratio stays within her own p05–p95, and the sibilance-band peak ratio ≤ her own p95.
   - Speech noise floor (`speech_noise_floor_db`) ≤ −60 dBFS. Compression raises breaths and noise, and ACX also uses −60.

   Whole-version checks: zero non-zero samples outside sentences; true peak ≤ −1.0 dBTP (4× oversampled); integrated LUFS within ±1 LU of the target. Report the sentence numbers that fell back.
3. **Final judgement by ear:** the app's existing ⑤ 鉴别 blind test.

### 7. Not verified / still to do
- The effect of EQ and compression on 像你本人 % has never been measured (step 6.1 measures it).
- Whether GPT-SoVITS V5's 48 kHz output has real content above 12–16 kHz.
- Whether pedalboard installs with `--no-deps` on the V5 numpy 1.26.4.
- Which filters the bundled 2021 ffmpeg.exe has.
- The exact AES TD1008, YouTube and Bilibili numbers.
- CPU time on her machine: not measured.

### Sources
**Standards and platform loudness**
- [Apple Podcasts audio requirements](https://podcasters.apple.com/support/audio-requirements)
- [EBU R128 s1](https://tech.ebu.ch/docs/r/r128s1.pdf), [EBU R128 (2014)](https://tech.ebu.ch/docs/r/r128-2014.pdf)
- [ACX audio submission requirements](https://help.acx.com/s/article/acx-audio-submission-requirements), [ACX blog: mastering audiobooks](https://www.acx.com/mp/blog/mastering-audiobooks-with-alex-the-audio-scientist)
- [Radio World on TD1008](https://www.radioworld.com/?p=115138), [Production Advice: TD1008](https://productionadvice.co.uk/td1008/)
- [Bilibili individual loudness test](https://www.bilibili.com/opus/942558919908130865)
- [Production Advice: YouTube stats for nerds](https://productionadvice.co.uk/stats-for-nerds/), [MeterPlugs: YouTube −14 LUFS](https://www.meterplugs.com/blog/2019/09/18/youtube-changes-loudness-reference-to-14-lufs.html)

**Voice EQ and processing practice**
- [iZotope podcast production guide, part 2](https://www.izotope.com/en/learn/podcast-production-guide-part2)
- [Sound On Sound: recording spoken word](https://www.soundonsound.com/techniques/recording-spoken-word), [Sound On Sound: Studio One, working with speech](https://www.soundonsound.com/techniques/studio-one-working-speech)
- [Avid: how to EQ vocals](https://www.avid.com/resource-center/how-to-eq-vocals)
- [Shure: reading a frequency response chart](https://www.shure.com/en-US/insights/how-to-read-a-microphone-frequency-response-chart), [Shure: recording and mixing vocals](https://www.shure.com/en-US/insights/how-to-record-and-mix-vocals)
- [Flypaper: EQing vocals](https://flypaper.soundfly.com/?p=20246)
- [Rode: audio processing for podcasting](https://rode.com/en/blog/all/audio-processing-and-fx-for-podcasting)
- [Bücklein 1981, JAES](https://www.aes.org/e-lib/browse.cfm?elib=10291)

**Tools**
- [pedalboard README](https://raw.githubusercontent.com/spotify/pedalboard/master/README.md), [pedalboard on PyPI](https://pypi.org/project/pedalboard/)
- [matchering README](https://raw.githubusercontent.com/sergree/matchering/master/README.md)
- [pyloudnorm on PyPI](https://pypi.org/project/pyloudnorm/)
- [timbral_models on PyPI](https://pypi.org/project/timbral_models/)
- [Auphonic single-track algorithms](https://auphonic.com/help/algorithms/singletrack.html)
- [Airwindows Consolidated](https://airwindows.com/consolidated/)

**Voice research**
- [LTAS in forensic phonetics (2023)](https://revistes.ub.edu/index.php/experimentalphonetics/article/view/44481), [Hughes et al., Interspeech 2017](https://eprints.whiterose.ac.uk/117386/1/Hughes_et_al_Interspeech_2017_.pdf)
- [Vocal attractiveness and body-size projection](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3634748/), [Chinese study on vocal attractiveness](https://www.sciengine.com/doi/pdf/7DB28FCA51D24843BC64651FF1E3EDEF)
- [Taiwanese speaking F0 study](https://wd.vghtpe.gov.tw/jcma/files/6704_179.pdf)

**Repo (read-only)**
- `/home/user/voicetwin/voicetwin/synth/engine.py`: `_levels`, `_render`, `_target_lufs`, `_make_variants`, `_version_score`, `denoise_light`, `PEAK_CEILING_DB`
- `/home/user/voicetwin/research/sv_benchmark/记录.md`
- `/home/user/voicetwin/voicetwin/eval/sv_frontend.py`
- `/home/user/voicetwin/research/gsv_v4_v5/官方资料/v5整合包/runtime_包版本.txt`
