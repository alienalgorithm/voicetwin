# 调研（美化）：ai-enhance（2026-11-24）

AI speech enhancement / restoration / "studio sound" for VoiceTwin: research report (research only, no files edited)

Network limit: arxiv.org and huggingface.co did not resolve from this session (WebFetch DNS failure). I read GitHub READMEs, code and PyPI directly. Every paper number below came through search-engine summaries of the papers and is labelled "(via search summary)". Nothing here was run on the teacher's audio.

## 1. Bottom line
- An enhancement model cannot make English pronunciation "more polished", and it should not. These models are built to keep the words exactly as they are, so any change to the words counts as a hallucination, which is a defect. Pronunciation has to come from the TTS side (reference choice, G2P, candidate selection by ASR). "Warmer / more magnetic" is also not what these models aim for: they push audio toward a generic "clean studio" balance. Whether that sounds warmer for her voice has to be measured.
- For the default "原声" version, do not use any generative model. Keep the existing rule (denoise only sentences whose speech-floor is above −60 dBFS). One possible upgrade: replace noisereduce with **DeepFilterNet3 using `atten_lim_db`**.
- For the optional "轻度美化" version, run a bake-off between two models:
  - **Sidon v0.1**: generative studio-quality restoration, 48 kHz, MIT code and weights, released 2025.
  - **DeepFilterNet3 with an attenuation limit**: conservative, phase-aligned, 48 kHz, MIT/Apache.
  - Resemble Enhance is the reserve generative candidate. Every sentence must pass a paired gate (section 5), or it falls back to the original.
- Bandwidth extension (AP-BWE / MossFormer2_SR_48K) only if a measurement shows the V5 output is band-limited. The project's own notes cite the official claim that the V5 vocoder "大幅降低高频混叠", so it is probably not needed. If it is used, keep the original signal below the crossover frequency.

## 2. Candidates
| Model | Kind | Output rate | Licence | Evidence | Verdict |
|---|---|---|---|---|---|
| DeepFilterNet3 (Rikorose) | Mask / deep filtering in the STFT domain. Output is phase-aligned (delay compensation is on by default). | 48 kHz only | MIT or Apache-2.0 | ClearerVoice SpeechScore table, VoiceBank+DEMAND at 48 kHz: PESQ 3.03, SI-SDR 15.71, **LSD 0.94 and LLR 0.55, the least spectral distortion of the 3 models tested**. README says Linux, macOS and Windows. | **Pick (conservative).** `atten_lim_db` is a built-in dry/wet mix: `enhanced = spec*lim + enh*(1-lim)` (df/enhance.py L238-240). Caveats: last PyPI release 0.5.6 was 2023-08-31. Compatibility with torch ≥2.7 on RTX 50 is unverified. |
| MossFormer2_SE_48K (Alibaba ClearerVoice-Studio) | Discriminative (mask-type is my reading; unverified) | 48 kHz | Apache-2.0 | Same table: PESQ 3.15, SI-SDR 19.36, OVRL 3.15, but **LSD 1.86, worse than the noisy input's 1.45**. The authors themselves note "anomalies in LLR and LSD with the 48 kHz models". | Alternative to DeepFilterNet3. The LSD anomaly is a timbre risk on clean input. |
| Sidon v0.1 (sarulab-speech) | Generative: w2v-BERT 2.0 features → vocoder (HiFi-GAN per paper, via search summary). It rebuilds the whole waveform. | 48 kHz | Code MIT; Sidon v0.1 checkpoint MIT (README) | README: quality "comparable to Miipher", up to 500× real time on one GPU, torch ≥2.8. Paper (via search summary): on FLEURS it beats Miipher-2 on CER and DNSMOS, is comparable on SpkSim, slightly worse on NISQA. LibriTTS test-clean SpkSim ≈0.971, DNSMOS ≈3.30 (from a garbled PDF extraction; unverified). | **Pick (generative polish), only behind the gate.** Its timing and phase are not those of the input, so it cannot be dry/wet mixed directly (section 4). README says the code is "only partially smoke-checked". Windows is not mentioned. |
| Resemble Enhance | Denoiser + latent CFM + vocoder (generative) | Fixed at 44.1 kHz (`hparams.py` asserts 44_100) | MIT | ClearerVoice table at 48 kHz: PESQ 2.84, SI-SDR 12.42, LSD 1.66 (worst PESQ and SI-SDR of the three). Hi-ResLDM paper (via search summary): Resemble Enhance best on DNSMOS. | Reserve only. Last PyPI release 2024-01-04. requirements.txt includes `deepspeed>=0.12.4`, which is hard to install on Windows. Output must be resampled back to 48 kHz. Knobs: `nfe`, `solver`, `lambd`, `tau` (stochastic, so fix the seed). |
| VoiceFixer | GAN + vocoder | 44.1 kHz | MIT | Hi-ResLDM paper (via search summary): VoiceFixer "altering the speaker's pitch and tone". | Reject. |
| AP-BWE | GAN that predicts amplitude and phase; bandwidth extension only (8/12/16/24 kHz → 48 kHz) | 48 kHz | MIT, code and weights | README: 292.3× real time on an RTX 4090, 18.1× on CPU. Trained on VCTK (English only). | Conditional, only if bandwidth is short. |
| MossFormer2_SR_48K | Super-resolution (bandwidth extension) | 48 kHz | Apache-2.0 | ClearVoice README: LSD from 16/24/32 kHz input goes 2.80→1.93, 2.60→1.52, 2.29→1.50. Model card: "designed for clean speech inputs". | Conditional (same condition). |
| StuPASE / PASE (Cisco, Interspeech 2026 / AAAI 2026) | Generative, built for low hallucination | **16 kHz** | Code Apache-2.0; weights licence not stated | Model card: DNSMOS 3.37, UTMOS 4.08, SIM 0.68, WER 11.57% on simulated noisy LibriSpeech | Reject: 16 kHz would throw away the 48 kHz band. Its finding is still useful (section 5). |
| SGMSE+ at 48 kHz (EARS-WHAM checkpoint) | Diffusion | 48 kHz | Code MIT; weights licence unverified | Built for noisy input; slow (multi-step sampling) | Reject for clean TTS output. |
| A2SB (NVIDIA) | Schrödinger-bridge bandwidth extension and inpainting, for music | 44.1 kHz | **NVIDIA licence, non-commercial only** | — | Reject (her Bilibili use). |
| Hi-ResLDM (Sony), VoiceBridge, AuraSE (24 kHz), NovaSR, FlashSR | — | — | — | No code or weights confirmed. github.com/sony/Hi-ResLDM returned 404 on 2026-11-24. NovaSR's claims are self-reported, with no benchmarks. | Not usable now. |
| Commercial: Adobe Podcast Enhance Speech v2 (Nov 2024; separate speech-strength slider since Feb 2025, per Adobe forum); NVIDIA Maxine Studio Voice ("High Quality" offline mode, 16/48 kHz) | Closed | — | Maxine needs an NVIDIA AI Enterprise licence (90-day evaluation available). RTX 50 support unverified. | — | Not integrable. Adobe would mean uploading her voice. Useful only as a listening reference. |

## 3. What the evidence does and does not show
- No source measures enhancement as a post-process on clean 48 kHz TTS output, with speaker similarity before and after. LibriTTS-R (Miipher) shows restoration helps when applied to *training data*, which is a different setup.
- The PASE/StuPASE authors (via search summary) report that DNSMOS and UTMOS **cannot detect hallucinations**, so speaker verification and ASR must sit next to any MOS predictor.
- On this project's own scale, from `research/sv_benchmark/记录.md` §10: "去杂音 + 绝对静音" moves 像你本人 by −2.5 ± 3.1 points; "音量放到最大" by −0.1 ± 0.9; sentences with only 1.5 s of voice by −13.3 ± 10.8. That is the measurement noise any gate has to respect.

## 4. Keeping it light and identity-preserving
- **Mask models (DeepFilterNet3, MossFormer2):** output stays sample-aligned, so lightness is a mix. Use `atten_lim_db` (e.g. a 6–12 dB cap; not yet tested) or a time-domain dry/wet ratio. After processing, cross-correlation lag must be 0.
- **Generative models (Sidon, Resemble):** a direct mix causes comb filtering. Two options I propose, both untested:
  - (a) Accept or reject whole sentences through the gate.
  - (b) STFT magnitude blend: |Y| = |X|^(1−a)·|E|^a, using the original phase. Align E to X first by cross-correlation. Choose a in about 0.3–0.5.
  - (c) An even safer variant: use the generative model only as an "oracle". Take the smoothed long-term spectrum ratio E/X over the whole lecture, cap it at ±2–3 dB, and apply it to the original as a fixed linear-phase EQ. This cannot hallucinate or change phonemes.
- **Bandwidth extension:** keep the original below the cutoff and add only the new highs through a linear-phase crossover. The band that carries identity and formants then stays untouched.
- Re-apply the existing per-sentence loudness match (`_match_levels`) and a true-peak limit after any processing.

## 5. Automatic verification (pair each sentence with its own original version)
1. **Identity:** a sentence's 像你本人 drop B vs A ≤ ~3 points, and the lecture mean drop ≤ ~1 point. The ~3 comes from the −2.5 ± 3.1 measured for a light denoise; the exact values are proposals to calibrate. Don't trust the gate on sentences with <2 s of voice; keep the original there.
   - Calibration trick: run each candidate on held-out *real* recordings of hers. 100% means her real recordings, so any drop there is pure identity drift caused by the processor.
2. **Content:** zero new ASR character errors per sentence (existing CER check). Add Whisper WER for English spans. Duration and sample count unchanged.
3. **Pitch, rate, timbre** (proposals):
   - F0 median shift ≤ ~0.3 semitone.
   - Speaking rate unchanged.
   - 1/3-octave long-term spectrum |Δ| ≤ 2–3 dB between 100 Hz and 8 kHz.
   - Report MCD against the original.
4. **Quality must actually improve:** Meta Audiobox Aesthetics PQ ("Production Quality"; CC-BY 4.0) should go up, UTMOSv2 (MIT) up, DNSMOS SIG/OVRL not down. Otherwise version B equals A for that sentence, and the UI says so honestly, as it already does. Avoid NISQA as the gate: its weights are CC BY-NC-SA 4.0.
5. **Artifacts:**
   - Floor inside within-sentence pauses ≤ the original's floor, using the existing `frame_rms_db` (generative vocoders can invent faint noise).
   - No clipping, no click or discontinuity.

## 6. Absolute digital silence
- Process each sentence before assembly, the way version B is built now. Pad each sentence with 0.2–0.5 s of zeros for the model, then crop back to the exact original length.
- Afterwards, re-impose the original's fades and endpoints (`out[0] = out[-1] = 0`, as `denoise_light` does). Then force exact zeros wherever the original sentence was exactly 0, and let the existing assembler insert exact-zero gaps.
- After Resemble Enhance, resample 44.1→48 kHz before this gating, so resampler ringing gets zeroed.
- Verify with the existing check that every gap sample == 0.

## 7. Suggested bake-off
- Use GitHub Actions or the teacher's PC; this dev machine cannot reach Hugging Face, per the handbook.
- Material: 30–50 held-out real sentences of hers plus their V5 syntheses.
- Candidates: DeepFilterNet3 at a few `atten_lim_db` values; Sidon as hard accept/reject, as magnitude blend at a = 0.3/0.5, and as oracle-EQ; Resemble Enhance at `lambd=0.1`, `tau=0.5`.
- Measure the effective bandwidth of the V5 output (energy above 16 kHz versus her real recordings) before considering bandwidth extension.
- Ship as the "美化" option only the variant that passes all gates on most sentences and raises PQ.

Unverified: MossFormer2 internals; Sidon's exact SpkSim/DNSMOS table; Windows / RTX 50 behaviour of Sidon, DeepFilterNet (PyPI 2023) and Resemble (deepspeed); all thresholds marked "proposal".

Sources:
- https://github.com/sarulab-speech/Sidon
- https://arxiv.org/abs/2509.17052
- https://www.researchgate.net/publication/395725305
- https://github.com/modelscope/ClearerVoice-Studio (clearvoice/README.md tables)
- https://github.com/resemble-ai/resemble-enhance (app.py, enhancer/inference.py, hparams.py, requirements.txt)
- https://github.com/Rikorose/DeepFilterNet (df/enhance.py)
- https://github.com/yxlu-0102/AP-BWE
- https://arxiv.org/abs/2409.11145 (Hi-ResLDM)
- https://github.com/cisco-open/pase
- https://huggingface.co/cisco-ai/stupase
- https://github.com/nvidia/diffusion-audio-restoration (LICENSE)
- https://github.com/sp-uhh/sgmse
- https://github.com/facebookresearch/audiobox-aesthetics
- https://github.com/sarulab-speech/UTMOSv2
- https://github.com/gabrielmittag/NISQA
- https://arxiv.org/abs/2601.13531 (ICASSP 2026 URGENT)
- https://arxiv.org/abs/2509.25275 (VoiceBridge)
- https://arxiv.org/pdf/2610.06632 (AuraSE)
- https://github.com/MaxMax2016/NovaSR
- https://community.adobe.com/t5/adobe-podcast-discussions/enhance-speech-v2-is-here/m-p/14992650
- https://docs.nvidia.com/maxine/afx/latest/AboutTheEffects/AboutStudioVoiceEffect.html
- https://forums.developer.nvidia.com/t/impossible-to-download-maxine-audio-effects-sdk/343193
- PyPI JSON for resemble-enhance, deepfilternet, clearvoice, voicefixer

Local files: /home/user/voicetwin/voicetwin/synth/engine.py (denoise_light L343, DENOISE_FLOOR_DB L506) and /home/user/voicetwin/research/sv_benchmark/记录.md §10. Downloaded READMEs are under /tmp/claude-0/-home-user/3d87cc6c-41dd-58c2-802f-d51dc0c78527/scratchpad/dl/.
