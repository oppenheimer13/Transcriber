# Transcriber

Long-form speech transcription with Whisper, from the command line.

A command-line tool that turns an audio or video recording into text. It is
built for hour-long material - meetings, interviews, talks - where the usual
"chop into 30-second windows" approach drifts, repeats itself, and hallucinates
`Thank you.` into every silent passage.

- **No ffmpeg install.** Audio is decoded in-process by PyAV, which ships
  ffmpeg's libraries inside the wheel. An `ffmpeg` on `PATH` is used only as a
  fallback for containers PyAV cannot open.
- **Swedish, properly.** KBLab's `kb-*` models are wired in and beat OpenAI's
  equivalents on Swedish by a wide margin.
- **Guards against hallucination.** Temperature fallback, a compression-ratio
  degeneracy check, and chunk planning that never pads a trailing sliver of
  speech out to a full window with silence.
- **Names come out right.** `--vocab` biases decoding toward the people,
  products and jargon in your recording.

## Install

Python 3.9 or newer. No ffmpeg, and no model downloads at install time.

**To use it** - `pipx` gives the tool its own isolated environment and puts a
`transcribe` command on your PATH, usable from any folder:

```powershell
py -m pip install --user pipx
pipx install git+https://github.com/oppenheimer13/Transcriber
```

**To work on it** - an editable install, so your edits take effect immediately:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -e .
```

Either way the dependencies come down automatically. `pip install -r
requirements.txt` still works and does the same as the editable install.

The optional backend is a separate, much larger install:
`pip install -e .[transformers]`, plus `.[vad]` for Silero VAD.

> Model weights are **not** installed. The one you pick is downloaded on first
> use and cached in `~/.cache/huggingface`, shared by every environment on the
> machine - so the first run of a given model is slow and the rest are instant.
> The default `large-v3-turbo` is 1.6 GB; `-m tiny` (75 MB) is a quick way to
> check an install works.

## Quick start

```powershell
transcribe meeting.m4a
```

Without installing, from a clone: `python -m transcriber meeting.m4a`.

Run with no `-m`, `-l` or `--vad` and it asks which model, which language and
whether to skip silence, then writes `meeting.txt` and `meeting.json` next to
the recording. With stdin closed (a scheduled task) each question takes its
default instead of stopping.

```powershell
# pick everything up front, no prompts
transcribe -m large-v3-turbo --vad off meeting.m4a

# Swedish interview, subtitles as well
transcribe -m kb-medium -l sv --formats txt,srt intervju.mp3

# several files in one load of the model. PowerShell does not expand globs
# for native commands, so let Get-ChildItem do it:
transcribe -m kb-large -l sv (gci *.m4a)
```

Progress, timings and warnings go to stderr; only the output files are written
to disk, so you can redirect one without losing the other.

## Models

`-m` takes any name below, or a full Hugging Face id (`KBLab/kb-whisper-large`).
Weights download on first use and are cached by `huggingface_hub`.

| Name | Language | Notes |
| --- | --- | --- |
| `tiny`, `base`, `small`, `medium` | multilingual | increasing size and accuracy |
| `large-v3` | multilingual | most accurate, slowest |
| `large-v3-turbo` | multilingual | **default** - near `large-v3` accuracy, much faster |
| `distil-large-v3` | English only | fast, English-only |
| `kb-tiny`, `kb-base`, `kb-small` | Swedish | `kb-small` beats OpenAI `small` by a wide margin |
| `kb-medium` | Swedish | better than `large-v3`, at a third the size |
| `kb-large` | Swedish | best available |

The `kb-*` repos carry CTranslate2 weights, so faster-whisper loads them
directly - no conversion step.

> Omit `-l` and you are asked. The offered default follows the model - `sv` for
> the `kb-*` models, `en` otherwise - so picking `kb-large` from the menu no
> longer quietly translates your Swedish into English. Pass `-l sv`, `-l auto`,
> or any other ISO code to skip the question.

## Output

`--formats` is a comma-separated list; the default is `txt,json`.

| Format | Contents |
| --- | --- |
| `txt` | flowing text with `[hh:mm:ss]` markers every minute (`--no-minute-markers` to omit) |
| `srt` | subtitles, `hh:mm:ss,mmm` |
| `vtt` | WebVTT subtitles |
| `json` | timestamped segments, full text, and the exact settings and timings of the run |

Files are named after the recording and land beside it, or in `-o <dir>`.
Note that `.gitignore` excludes `*.txt`, `*.json`, `*.srt` and `*.vtt` so
transcripts never get committed by accident.

## Vocabulary

Whisper mangles names it has never seen. Give it a list and they come out right:

```powershell
transcribe --vocab "EPAM, AstraZeneca, Kubernetes" meeting.m4a
transcribe --vocab @terms.txt meeting.m4a
```

With no `--vocab`, it looks for `<recording>.vocab.txt` and then
`vocab.txt` beside the audio, and uses the first it finds. One term per line;
`#` starts a comment. Terms are passed as faster-whisper *hotwords*, so they
bias every window rather than just the first - this needs the faster-whisper
backend and is ignored elsewhere.

## Options

| Flag | Default | What it does |
| --- | --- | --- |
| `-m`, `--model` | ask, else `large-v3-turbo` | model name or Hugging Face id |
| `-l`, `--language` | ask, else `sv` for `kb-*` / `en` | ISO code, or `auto` to detect |
| `--task` | `transcribe` | or `translate`, to English |
| `--backend` | `auto` | `faster-whisper` or `transformers` |
| `--device` | `auto` | `cpu` or `cuda` |
| `--compute-type` | `auto` | faster-whisper precision: `int8`, `float16`, `float32`, ... |
| `--beams` | `5` | `1` is fastest, `5` more accurate |
| `--vad` | ask, else `off` | drop non-speech before decoding |
| `--vad-min-silence` | `400` | ms of silence that splits speech |
| `--vocab` | auto-detect | terms to bias toward, inline or `@file` |
| `--condition-on-previous` | `off` | feed earlier text back as context |
| `--formats` | `txt,json` | `txt`, `srt`, `vtt`, `json` |
| `-o`, `--output-dir` | beside the input | where to write |
| `--no-minute-markers` | off | omit `[hh:mm:ss]` markers from the `.txt` |

`--mode`, `--batch-size`, `--chunk-s` and `--overlap-s` apply to the
transformers backend only; `transcribe -h` covers them. `--chunk-s`
cannot exceed 30 - Whisper's receptive field is fixed at that length, and a
longer window would be truncated with the remainder silently discarded.

### Silence filtering

`--vad off` is the default and the right choice for meetings: every quiet
passage is still decoded, so a soft-spoken participant is not dropped.
`--vad on` skips silence and runs faster, at the risk of clipping quiet
speakers. `--vad auto` means on for faster-whisper, off for transformers.

### Backends

`auto` picks **faster-whisper** whenever it is installed and the model has a
CTranslate2 build - roughly 4x faster than transformers on CPU at the same
accuracy, with a progress bar and an ETA in realtime multiples.

**transformers** is the fallback, in two modes. `longform` (default) uses
Whisper's native sequential long-form decoding. `chunked` windows the audio
explicitly, optionally cutting on silence with Silero VAD (`--vad on`, which
needs the `vad` extra), and de-duplicates the overlap between windows.

## Troubleshooting

**It says "produced no text".** Usually `--vad on` on a quiet recording. Rerun
with `--vad off`.

**The transcript is in English but the audio isn't.** The language question
took its default. Pass `-l sv` (or the right ISO code, or `-l auto`) explicitly
- this happens most easily in a scheduled task, where every prompt is answered
by its default.

**"PyAV could not decode".** It falls back to an `ffmpeg` on `PATH` if
there is one; otherwise convert the file to `.wav` first.

**`OMP: Error #15`.** torch and ctranslate2 each bundle an Intel OpenMP
runtime. Transcriber avoids importing both, so this points at something else
in the environment loading torch first - a `pipx` install cannot hit this,
since it gets an environment of its own.

**It looks hung on a long file.** Transformers long-form decoding exposes no
per-window hook, so it prints an elapsed-time heartbeat every minute instead of
a bar. The default faster-whisper backend shows a real progress bar.

**`transcribe` is not recognised.** The `pipx` install puts it on your PATH, but
an open terminal will not pick that up until you restart it. `pipx ensurepath`
fixes a PATH that never got set up.

## License

MIT - see [LICENSE](LICENSE).
