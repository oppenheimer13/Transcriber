#!/usr/bin/env python3
"""
transcribe - long-form speech transcription with Whisper.

Backends
    faster-whisper  CTranslate2 build, roughly 4x faster than transformers on
                    CPU at equal accuracy. Used automatically when installed.
    transformers    Hugging Face Whisper. Two modes:
                      longform (default) - native sequential long-form decoding
                                           with temperature fallback and real
                                           segment timestamps
                      chunked            - explicit windowing, optionally cut on
                                           silence with Silero VAD

Install
    pipx install git+https://github.com/oppenheimer13/Transcriber
    pip install -e .[transformers]    # only for --backend transformers
    pip install -e .[vad]             # only for --vad on with that backend
    Decoding needs no external ffmpeg: PyAV bundles the ffmpeg libraries.
    An ffmpeg on PATH is used as a fallback. Model weights are downloaded on
    first use, not at install time. See README.md for the full guide.

Examples
    transcribe meeting.m4a
    transcribe -m large-v3-turbo --formats txt,srt talk.wav
    transcribe -m kb-medium -l sv intervju.mp3
    transcribe --vad off meeting.m4a          # keep every quiet passage
    transcribe --backend transformers --mode chunked --vad on x.mp3

Model, language and silence filtering are asked for interactively when -m, -l
or --vad is omitted, so a bare `transcribe file.m4a` is enough.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
import os
import shutil
import subprocess
import sys
import time
import zlib
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from . import __version__

SCRIPT_VERSION = __version__  # recorded in the JSON metadata of every run
SR = 16000
WHISPER_WINDOW_S = 30.0  # Whisper's fixed receptive field

# short name -> (hugging face id, faster-whisper id or None if unavailable)
MODELS = {
    "tiny": ("openai/whisper-tiny", "tiny"),
    "base": ("openai/whisper-base", "base"),
    "small": ("openai/whisper-small", "small"),
    "medium": ("openai/whisper-medium", "medium"),
    "large-v3": ("openai/whisper-large-v3", "large-v3"),
    "large-v3-turbo": ("openai/whisper-large-v3-turbo", "large-v3-turbo"),
    "distil-large-v3": ("distil-whisper/distil-large-v3", "distil-large-v3"),
    # KBLab's Swedish-tuned models. The same repo carries CTranslate2 weights,
    # so faster-whisper loads these ids directly - no conversion needed.
    "kb-tiny": ("KBLab/kb-whisper-tiny", "KBLab/kb-whisper-tiny"),
    "kb-base": ("KBLab/kb-whisper-base", "KBLab/kb-whisper-base"),
    "kb-small": ("KBLab/kb-whisper-small", "KBLab/kb-whisper-small"),
    "kb-medium": ("KBLab/kb-whisper-medium", "KBLab/kb-whisper-medium"),
    "kb-large": ("KBLab/kb-whisper-large", "KBLab/kb-whisper-large"),
}

TEMPERATURE_FALLBACK = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
COMPRESSION_RATIO_LIMIT = 2.4  # text-space, openai-whisper convention


# shown by the interactive picker, in menu order
MODEL_MENU = [
    ("tiny", "fastest, roughest"),
    ("base", ""),
    ("small", ""),
    ("medium", ""),
    ("large-v3", "most accurate, slowest"),
    ("large-v3-turbo", "default - near large-v3 accuracy, much faster"),
    ("distil-large-v3", "English only"),
    ("kb-small", "Swedish - beats OpenAI small by a wide margin"),
    ("kb-medium", "Swedish - better than large-v3, a third the size"),
    ("kb-large", "Swedish - best available"),
]
DEFAULT_MODEL = "large-v3-turbo"

VAD_MENU = [
    ("off", "recommended for meetings - decodes every quiet passage"),
    ("on", "skips silence, so it runs faster, but can clip quiet speakers"),
    ("auto", "on for faster-whisper, off for transformers"),
]
DEFAULT_VAD = "off"

# shown by the interactive picker; any other ISO code can still be typed
LANGUAGE_MENU = [
    ("en", "English"),
    ("sv", "Swedish"),
    ("auto", "detect from the audio"),
]


@dataclass
class Segment:
    start: float
    end: float
    text: str


def _prompt_menu(title: str, label: str, entries: Sequence[Tuple[str, str]],
                 default: str, also_accept: Sequence[str] = (),
                 allow_hub_id: bool = False, allow_lang_code: bool = False) -> str:
    """Numbered picker on stderr. Falls through to `default` when not a TTY."""
    if not sys.stdin.isatty():
        return default

    print(title, file=sys.stderr)
    for i, (name, note) in enumerate(entries, 1):
        suffix = f"  ({note})" if note else ""
        print(f"  {i:>2}) {name}{suffix}", file=sys.stderr)

    valid = {name for name, _ in entries} | {str(v) for v in also_accept}
    while True:
        try:
            choice = input(f"{label} [{default}], or q to quit: ").strip()
        except EOFError:
            # stdin is closed or exhausted - a scheduled task, a pipeline, a
            # terminal that reports isatty() but has nothing to read. Falling
            # back to the default keeps unattended runs working; aborting the
            # batch here would be a worse answer than the documented default.
            print(f"\n  no input available - using '{default}'", file=sys.stderr)
            return default
        except KeyboardInterrupt:
            print("\nCancelled.", file=sys.stderr)
            raise SystemExit(130)
        if not choice:
            return default
        if choice.lower() in ("q", "quit", "exit"):
            print("Cancelled.", file=sys.stderr)
            raise SystemExit(130)
        if choice.isdigit() and 1 <= int(choice) <= len(entries):
            return entries[int(choice) - 1][0]
        if allow_hub_id and "/" in choice:
            return choice  # raw hub id - case matters
        if allow_lang_code and choice.isalpha() and 2 <= len(choice) <= 3:
            return choice.lower()  # any ISO code, not just the menu entries
        if choice.lower() in valid:
            return choice.lower()
        print(f"  '{choice}' is not on the list - pick a number or a name.",
              file=sys.stderr)


def prompt_for_model(default: str = DEFAULT_MODEL) -> str:
    return _prompt_menu("Available models:", "Select model", MODEL_MENU,
                        default, also_accept=MODELS, allow_hub_id=True)


def prompt_for_vad(default: str = DEFAULT_VAD) -> str:
    return _prompt_menu("Silence filtering:", "Select", VAD_MENU, default)


def default_language(model_name: str) -> str:
    """KBLab's models are Swedish-tuned; anything else defaults to English.

    Getting this wrong is silent and total: with -l en a Swedish recording is
    translated rather than transcribed, and the output looks plausible.
    """
    return "sv" if model_name.lower().startswith(("kb-", "kblab/")) else "en"


def prompt_for_language(default: str = "en") -> str:
    return _prompt_menu("Language:", "Select language", LANGUAGE_MENU,
                        default, allow_lang_code=True)


# --------------------------------------------------------------------------- #
# audio
# --------------------------------------------------------------------------- #

def _decode_av(path: str, sr: int) -> np.ndarray:
    """Decode in-process with PyAV, which bundles ffmpeg's libraries."""
    import av  # noqa: PLC0415

    with av.open(path) as container:
        if not container.streams.audio:
            raise RuntimeError(f"no audio stream in {path}")
        stream = container.streams.audio[0]
        stream.thread_type = "AUTO"
        resampler = av.AudioResampler(format="s16", layout="mono", rate=sr)

        blocks: List[np.ndarray] = []
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                blocks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):  # flush the resampler's tail
            blocks.append(out.to_ndarray().reshape(-1))

    if not blocks:
        return np.zeros(0, np.float32)
    return np.concatenate(blocks).astype(np.float32) / 32768.0


def _decode_ffmpeg_cli(path: str, sr: int) -> np.ndarray:
    """Decode by piping s16le out of the ffmpeg binary on PATH."""
    cmd = [
        "ffmpeg", "-nostdin", "-threads", "0",
        "-i", path,
        "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le", "-ar", str(sr),
        "-",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", "replace").strip().splitlines()[-6:]
        raise RuntimeError(
            f"ffmpeg could not decode {path}:\n  " + "\n  ".join(tail)
        )
    return np.frombuffer(proc.stdout, np.int16).astype(np.float32) / 32768.0


def load_audio(path: str, sr: int = SR) -> np.ndarray:
    """Decode any container to mono float32 at `sr`.

    PyAV does the work: it ships ffmpeg's libraries inside the wheel, so a
    plain `pip install -r requirements.txt` is enough and the decode behaves
    the same on every machine. The ffmpeg CLI stays as a fallback for the
    rare container PyAV's build cannot open; the two agree sample for sample.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    try:
        return _decode_av(path, sr)
    except Exception as exc:  # noqa: BLE001 - any decode failure is worth retrying
        if not shutil.which("ffmpeg"):
            # PyAV's errors subclass ValueError, not RuntimeError, so re-raise
            # as RuntimeError: callers should not have to know which decoder
            # ran to catch a bad file.
            raise RuntimeError(f"could not decode {path}: {exc}") from exc
        print(f"  note: PyAV could not decode ({exc}); using the ffmpeg CLI",
              file=sys.stderr)
        return _decode_ffmpeg_cli(path, sr)


# --------------------------------------------------------------------------- #
# text helpers
# --------------------------------------------------------------------------- #

def format_timestamp(seconds: float, decimal: str = ".", ms: bool = False) -> str:
    seconds = max(0.0, float(seconds))
    if not ms:
        h, rem = divmod(int(seconds), 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"
    # Round to whole milliseconds *before* splitting. Splitting first and then
    # rounding the fraction lets it carry to 1000 and emit a 4-digit field:
    # 1.9996 -> "00:00:01,1000", which is not valid SRT or WebVTT.
    total_ms = int(round(seconds * 1000))
    h, rem = divmod(total_ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, thousandths = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{decimal}{thousandths:03d}"


def compression_ratio(text: str) -> float:
    """Degeneracy detector: looping output compresses far better than speech."""
    data = text.encode("utf-8")
    if not data:
        return 0.0
    return len(data) / len(zlib.compress(data))


def dedup_overlap_words(
    context: str,
    new_text: str,
    max_tail_words: int = 60,
    min_match_words: int = 5,
) -> str:
    """Trim a duplicated prefix from `new_text` caused by audio overlap.

    Finds the longest k where the last k words of `context` equal the first k
    words of `new_text` (case- and punctuation-insensitive) and drops them.
    """
    if not context or not new_text:
        return new_text

    ctx_words = context.split()
    new_words = new_text.split()
    if not ctx_words or not new_words:
        return new_text

    def norm(w: str) -> str:
        return "".join(c for c in w.lower() if c.isalnum())

    tail = [norm(w) for w in ctx_words[-max_tail_words:]]
    head = [norm(w) for w in new_words]

    for k in range(min(len(tail), len(head)), min_match_words - 1, -1):
        if tail[-k:] == head[:k]:
            return " ".join(new_words[k:]).lstrip()
    return new_text


# --------------------------------------------------------------------------- #
# chunk planning
# --------------------------------------------------------------------------- #

def plan_fixed_chunks(
    n_samples: int,
    sr: int,
    chunk_s: float,
    overlap_s: float,
    min_tail_s: float = 0.25,
) -> List[Tuple[int, int]]:
    """Overlapping windows that never end on a silence-padded sliver.

    The old script looped `range(0, len(audio), hop)`, so the last window could
    hold a fraction of a second of new speech padded out to 30s with silence -
    reliably hallucinated into "Thank you." / "Subtitles by ...".

    Here the final window is always slid backwards to end at EOF at full length,
    whenever the recording is long enough to hold one. The extra overlap costs
    one window of compute and is removed again by `dedup_overlap_words`.
    """
    if n_samples <= 0:
        return []

    step = int(round(chunk_s * sr))
    overlap = int(round(overlap_s * sr))
    if step <= 0:
        raise ValueError("chunk_s must be positive")
    if overlap >= step:
        raise ValueError("overlap_s must be smaller than chunk_s")
    hop = step - overlap
    min_tail = int(round(min_tail_s * sr))

    windows: List[Tuple[int, int]] = []
    start = 0
    while True:
        end = min(start + step, n_samples)
        windows.append((start, end))
        if end >= n_samples:
            break

        nxt = start + hop
        if nxt + step >= n_samples:  # this would be the final window
            new_audio = n_samples - end
            if new_audio < min_tail:
                break  # trailing sliver, almost certainly silence
            # Whisper pads anything shorter than its 30s field with silence,
            # and that padding is what provokes "Thank you." / "Subtitles by
            # ..." hallucinations. A `new_audio < min_new` test only caught the
            # very smallest tails: 2.5s of new audio still produced a 7.5s
            # window padded with 22.5s of silence. Whenever a full-length
            # window fits at all, slide back to one. The extra overlap costs a
            # single window of compute and is removed by dedup_overlap_words.
            if n_samples >= step:
                nxt = n_samples - step
        if nxt <= start:
            break
        start = nxt
    return windows


def plan_vad_chunks(
    audio: np.ndarray,
    sr: int,
    max_chunk_s: float,
    pad_s: float = 0.2,
    min_silence_ms: int = 400,
) -> List[Tuple[int, int]]:
    """Cut on silence instead of on the clock, so no word straddles a boundary.

    Silence is only a *preferred* boundary, never a guaranteed one: a speaker
    who runs on for minutes without a `min_silence_ms` pause yields a single
    speech region far longer than `max_chunk_s`. Whisper's field is fixed at 30s
    and the feature extractor truncates anything longer without a word, so such
    a region is hard-split rather than allowed through - a cut mid-word costs a
    garbled word, letting it through costs everything past the 30s mark.
    """
    try:
        from silero_vad import get_speech_timestamps, load_silero_vad  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "--vad needs the silero-vad package:  pip install silero-vad"
        ) from exc

    import torch  # noqa: PLC0415

    vad = load_silero_vad()
    speech = get_speech_timestamps(
        torch.from_numpy(audio),
        vad,
        sampling_rate=sr,
        min_silence_duration_ms=min_silence_ms,
    )
    if not speech:
        return []

    limit = int(max_chunk_s * sr)
    pad = int(pad_s * sr)
    windows: List[Tuple[int, int]] = []

    def emit(start: int, end: int) -> None:
        """Append, hard-splitting anything the 30s field cannot hold."""
        while end - start > limit:
            windows.append((start, start + limit))
            start += limit
        windows.append((start, end))

    cur_start = speech[0]["start"]
    cur_end = speech[0]["end"]

    for region in speech[1:]:
        if region["end"] - cur_start > limit:
            emit(cur_start, cur_end)
            cur_start, cur_end = region["start"], region["end"]
        else:
            cur_end = region["end"]
    emit(cur_start, cur_end)

    padded: List[Tuple[int, int]] = []
    for s, e in windows:
        s = max(0, s - pad)
        e = min(len(audio), e + pad)
        # the padding itself must not push a window back over the limit
        padded.append((s, min(e, s + limit)))
    return padded


# --------------------------------------------------------------------------- #
# backends
# --------------------------------------------------------------------------- #

def _progress(iterable, total=None, desc=""):
    try:
        from tqdm import tqdm  # noqa: PLC0415

        return tqdm(iterable, total=total, desc=desc)
    except ImportError:
        return iterable


def _audio_bar(total_s: float):
    """Progress measured in seconds of audio, so the ETA is meaningful."""
    try:
        from tqdm import tqdm  # noqa: PLC0415
    except ImportError:
        return None
    return tqdm(
        total=max(1.0, total_s),
        bar_format=("Transcribing: {percentage:3.0f}%|{bar}| "
                    "{n:.0f}/{total:.0f}s audio [{elapsed}<{remaining}, {postfix}]"),
        postfix="estimating",
    )


def _dtype_kwarg(dtype) -> dict:
    """transformers renamed `torch_dtype` to `dtype` in 4.56."""
    import transformers  # noqa: PLC0415

    try:
        version = tuple(int(part) for part in transformers.__version__.split(".")[:2])
    except ValueError:
        return {"dtype": dtype}
    return {"dtype": dtype} if version >= (4, 56) else {"torch_dtype": dtype}


class Heartbeat:
    """Native long-form decoding exposes no per-window hook, so print elapsed
    time rather than leaving a long file looking hung."""

    def __init__(self, interval: float = 60.0, stream=sys.stderr):
        self.interval = interval
        self.stream = stream
        self._stop = None
        self._thread = None
        self._started = 0.0

    def __enter__(self):
        import threading  # noqa: PLC0415
        import time  # noqa: PLC0415

        self._started = time.monotonic()
        self._stop = threading.Event()

        def tick():
            while not self._stop.wait(self.interval):
                mins = (time.monotonic() - self._started) / 60.0
                print(f"  still decoding, {mins:.0f} min elapsed",
                      file=self.stream, flush=True)

        self._thread = threading.Thread(target=tick, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        import time  # noqa: PLC0415

        if self._stop is not None:
            self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if exc[0] is None:
            mins = (time.monotonic() - self._started) / 60.0
            print(f"  decoded in {mins:.1f} min", file=self.stream, flush=True)
        return False


class FasterWhisperBackend:
    name = "faster-whisper"

    def __init__(self, model_id: str, device: str, compute_type: str,
                 language: Optional[str], task: str, beams: int, vad: bool,
                 vad_min_silence: int = 400, condition_on_previous: bool = True,
                 vocab: Sequence[str] = ()):
        from faster_whisper import WhisperModel  # noqa: PLC0415

        self.language = language
        self.task = task
        self.beams = beams
        self.vad = vad
        self.vad_min_silence = vad_min_silence
        self.condition_on_previous = condition_on_previous
        self.vocab = list(vocab)
        self.model = WhisperModel(model_id, device=device, compute_type=compute_type)

    def transcribe(self, audio: np.ndarray,
                   vocab: Sequence[str] = ()) -> List[Segment]:
        terms = list(vocab) or self.vocab
        kwargs = dict(
            language=self.language,
            task=self.task,
            beam_size=self.beams,
            temperature=list(TEMPERATURE_FALLBACK),
            compression_ratio_threshold=COMPRESSION_RATIO_LIMIT,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            condition_on_previous_text=self.condition_on_previous,
            vad_filter=self.vad,
            vad_parameters={"min_silence_duration_ms": self.vad_min_silence},
        )
        if terms:
            # hotwords bias every window, unlike initial_prompt which only
            # reaches the first one when conditioning is off
            kwargs["hotwords"] = ", ".join(terms)

        try:
            segments, info = self.model.transcribe(audio, **kwargs)
        except TypeError:
            kwargs.pop("hotwords", None)
            print("  note: this faster-whisper version has no hotword support; "
                  "--vocab ignored", file=sys.stderr)
            segments, info = self.model.transcribe(audio, **kwargs)
        # With VAD on, only speech is decoded, so measure against that.
        total = getattr(info, "duration_after_vad", None) or info.duration
        bar = _audio_bar(total)
        started = time.monotonic()
        position = 0.0

        out: List[Segment] = []
        for s in segments:
            text = s.text.strip()
            if text:
                out.append(Segment(float(s.start), float(s.end), text))
            if bar is not None:
                position = max(position, min(float(s.end), total))
                bar.n = position
                elapsed = time.monotonic() - started
                if elapsed > 1:
                    bar.postfix = f"{float(s.end) / elapsed:.1f}x realtime"
                bar.refresh()
        if bar is not None:
            bar.n = bar.total
            bar.refresh()
            bar.close()
        return out


class TransformersBackend:
    name = "transformers"

    def __init__(self, model_id: str, device: str, language: Optional[str],
                 task: str, beams: int, batch_size: int,
                 condition_on_previous: bool = True):
        import torch  # noqa: PLC0415
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor  # noqa: PLC0415

        self.torch = torch
        self.device = device
        self.dtype = torch.float16 if device == "cuda" else torch.float32
        self.language = language
        self.task = task
        self.beams = beams
        self.batch_size = batch_size
        self.condition_on_previous = condition_on_previous

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, low_cpu_mem_usage=True, **_dtype_kwarg(self.dtype)
        ).to(device)
        self.model.eval()

    # -- shared ------------------------------------------------------------ #

    def _lang_kwargs(self) -> dict:
        kw = {"task": self.task}
        if self.language:
            kw["language"] = self.language
        return kw

    # -- native long-form -------------------------------------------------- #

    def transcribe_longform(self, audio: np.ndarray) -> List[Segment]:
        duration = len(audio) / SR
        if duration < WHISPER_WINDOW_S:
            return self._transcribe_short(audio, duration)

        inputs = self.processor(
            audio,
            sampling_rate=SR,
            return_tensors="pt",
            truncation=False,
            padding="longest",
            return_attention_mask=True,
        ).to(self.device, self.dtype)

        with self.torch.no_grad(), Heartbeat():
            out = self.model.generate(
                **inputs,
                **self._lang_kwargs(),
                return_timestamps=True,
                return_segments=True,
                num_beams=self.beams,
                # Whisper's own anti-loop mechanism: retry a window at a higher
                # temperature when it looks degenerate. This replaces
                # repetition_penalty / no_repeat_ngram_size, which forbid
                # legitimate repetition and raise WER.
                temperature=TEMPERATURE_FALLBACK,
                compression_ratio_threshold=1.35,  # token-space in transformers
                logprob_threshold=-1.0,
                no_speech_threshold=0.6,
                condition_on_prev_tokens=self.condition_on_previous,
            )

        segments: List[Segment] = []
        for seg in out["segments"][0]:
            text = self.processor.tokenizer.decode(
                seg["tokens"], skip_special_tokens=True
            ).strip()
            if text:
                segments.append(
                    Segment(float(seg["start"]), float(seg["end"]), text)
                )
        return segments

    def _transcribe_short(self, audio: np.ndarray, duration: float) -> List[Segment]:
        inputs = self.processor(
            audio, sampling_rate=SR, return_tensors="pt", return_attention_mask=True
        ).to(self.device, self.dtype)
        with self.torch.no_grad():
            ids = self.model.generate(
                **inputs, **self._lang_kwargs(),
                return_timestamps=True, num_beams=self.beams,
            )
        decoded = self.processor.batch_decode(
            ids, skip_special_tokens=True, output_offsets=True
        )[0]
        segments = [
            Segment(
                float(o["timestamp"][0]),
                float(o["timestamp"][1]) if o["timestamp"][1] is not None else duration,
                o["text"].strip(),
            )
            for o in decoded.get("offsets", [])
            if o["text"].strip()
        ]
        if not segments and decoded["text"].strip():
            segments = [Segment(0.0, duration, decoded["text"].strip())]
        return segments

    # -- explicit chunking ------------------------------------------------- #

    def _decode(self, clips: Sequence[np.ndarray], temperature: float) -> List[str]:
        inputs = self.processor(
            list(clips), sampling_rate=SR, return_tensors="pt",
            return_attention_mask=True,
        ).to(self.device, self.dtype)

        kwargs = dict(self._lang_kwargs())
        if temperature > 0:
            kwargs.update(do_sample=True, temperature=temperature, num_beams=1)
        else:
            kwargs.update(do_sample=False, num_beams=self.beams)

        with self.torch.no_grad():
            ids = self.model.generate(**inputs, **kwargs)
        return [t.strip() for t in self.processor.batch_decode(ids, skip_special_tokens=True)]

    def transcribe_chunked(
        self, audio: np.ndarray, windows: Sequence[Tuple[int, int]],
        dedup: bool = True,
    ) -> List[Segment]:
        segments: List[Segment] = []
        context = ""  # bounded rolling context, not the whole transcript

        batches = [
            windows[i:i + self.batch_size]
            for i in range(0, len(windows), self.batch_size)
        ]
        prev_window_end = 0  # samples; used to size the de-duplication window
        for batch in _progress(batches, total=len(batches), desc="Transcribing"):
            clips = [audio[s:e] for s, e in batch]
            texts = self._decode(clips, 0.0)

            for idx, ((start, end), text) in enumerate(zip(batch, texts)):
                # per-clip temperature fallback for degenerate output
                if text and compression_ratio(text) > COMPRESSION_RATIO_LIMIT:
                    best = text
                    for temp in TEMPERATURE_FALLBACK[1:]:
                        retry = self._decode([clips[idx]], temp)[0]
                        if retry and compression_ratio(retry) <= COMPRESSION_RATIO_LIMIT:
                            best = retry
                            break
                        if retry and compression_ratio(retry) < compression_ratio(best):
                            best = retry
                    text = best

                if not text:
                    continue
                if dedup:
                    # Size the comparison to the overlap actually present. The
                    # final window is slid back to full length, which can leave
                    # a ~29s overlap - far more than a fixed 60-word tail can
                    # see, so the duplicate survived into the transcript.
                    # ~3 words/s of speech, doubled for headroom.
                    overlap_s = max(0.0, (prev_window_end - start) / SR)
                    tail_words = max(60, int(overlap_s * 6))
                    text = dedup_overlap_words(context, text,
                                               max_tail_words=tail_words)
                else:
                    tail_words = 60
                if not text:
                    continue

                # Window bounds, not where the text sits: with overlap every cue
                # would otherwise start before the previous one ended, which is
                # invalid in SRT and WebVTT. Without word timestamps the honest
                # approximation is to butt each cue against its predecessor.
                start_s, end_s = start / SR, end / SR
                if segments and start_s < segments[-1].end:
                    start_s = segments[-1].end
                if end_s <= start_s:
                    end_s = start_s + 0.001  # keep cues strictly increasing
                segments.append(Segment(start_s, end_s, text))

                prev_window_end = max(prev_window_end, end)
                # the context must be able to hold everything dedup will scan
                keep = max(80, tail_words + 40)
                context = " ".join((context + " " + text).split()[-keep:])
        return segments


# --------------------------------------------------------------------------- #
# writers
# --------------------------------------------------------------------------- #

def write_txt(segments: Sequence[Segment], path: str, minute_markers: bool = True) -> None:
    parts: List[str] = []  # list + join; += in a loop is O(n^2)
    next_marker = 60.0
    for seg in segments:
        if minute_markers:
            while seg.start >= next_marker:
                parts.append(f"\n\n[{format_timestamp(next_marker)}]\n")
                next_marker += 60.0
        parts.append(seg.text + " ")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("".join(parts).strip() + "\n")


def write_srt(segments: Sequence[Segment], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for i, seg in enumerate(segments, 1):
            fh.write(
                f"{i}\n"
                f"{format_timestamp(seg.start, ',', ms=True)} --> "
                f"{format_timestamp(seg.end, ',', ms=True)}\n"
                f"{seg.text}\n\n"
            )


def write_vtt(segments: Sequence[Segment], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("WEBVTT\n\n")
        for seg in segments:
            fh.write(
                f"{format_timestamp(seg.start, '.', ms=True)} --> "
                f"{format_timestamp(seg.end, '.', ms=True)}\n"
                f"{seg.text}\n\n"
            )


def write_json(segments: Sequence[Segment], path: str, meta: dict) -> None:
    payload = {
        **meta,
        "segments": [
            {"start": round(s.start, 3), "end": round(s.end, 3), "text": s.text}
            for s in segments
        ],
        "text": " ".join(s.text for s in segments).strip(),
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


WRITERS = {"txt": write_txt, "srt": write_srt, "vtt": write_vtt, "json": write_json}


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #

def resolve_model(name: str, backend: str) -> str:
    if "/" in name:  # raw hub id
        return name
    if name not in MODELS:
        raise SystemExit(
            f"Unknown model '{name}'. Choose one of: {', '.join(MODELS)} "
            "or pass a full Hugging Face id."
        )
    hf_id, ct2_id = MODELS[name]
    if backend == "faster-whisper":
        if ct2_id is None:
            raise SystemExit(
                f"'{name}' has no CTranslate2 build; run it with "
                "--backend transformers."
            )
        return ct2_id
    return hf_id


def pick_backend(requested: str, model: str) -> str:
    if requested != "auto":
        return requested
    ct2_available = "/" in model or MODELS.get(model, (None, None))[1] is not None
    if ct2_available:
        try:
            import faster_whisper  # noqa: F401,PLC0415

            return "faster-whisper"
        except ImportError:
            pass
    return "transformers"


def load_vocab(spec: str) -> List[str]:
    """Vocabulary terms, given inline or as @path to a text file.

    Returns a list rather than a joined string so that a term containing a
    comma ("Smith, John") stays one term - joining first and splitting later
    counted it as two.

    A file that cannot be read or decoded raises RuntimeError, so a batch can
    skip the one recording it belongs to.
    """
    if not spec:
        return []
    if spec.startswith("@"):
        path = spec[1:]
        try:
            with open(path, "rb") as fh:
                data = fh.read()
        except OSError as exc:
            raise RuntimeError(f"could not read vocabulary file: {exc}") from exc
        try:
            # utf-8-sig: Notepad and PowerShell often write a BOM, which would
            # otherwise end up glued to the first term
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            # Notepad's "ANSI" encoding stores ö and å as single bytes that are
            # not valid UTF-8. This is a ValueError, so the OSError handler
            # above never saw it and the run ended in a traceback.
            lineno = exc.object[:exc.start].count(b"\n") + 1
            raise RuntimeError(f"vocabulary file {path} is not UTF-8 (line "
                               f"{lineno}); re-save it with UTF-8 encoding") from exc
        # strip before testing for '#': an indented comment is still a
        # comment, and was otherwise imported as a vocabulary term
        lines = (line.strip() for line in text.splitlines())
        return [ln for ln in lines if ln and not ln.startswith("#")]
    return [t.strip() for t in spec.split(",") if t.strip()]


def find_vocab(audio_path: str) -> Optional[str]:
    """Look for a vocabulary file to go with this recording.

    <recording>.vocab.txt wins, then vocab.txt in the same folder.
    """
    stem = os.path.splitext(os.path.abspath(audio_path))[0]
    folder = os.path.dirname(os.path.abspath(audio_path))
    for candidate in (stem + ".vocab.txt", os.path.join(folder, "vocab.txt")):
        if os.path.isfile(candidate):
            return candidate
    return None


def pick_device(requested: str, backend: str = "transformers") -> str:
    if requested != "auto":
        return requested
    if backend == "faster-whisper":
        # Do NOT import torch here. torch and ctranslate2 each bundle their own
        # Intel OpenMP runtime, and loading both aborts the process with
        # "OMP: Error #15". ctranslate2 can answer this question itself.
        try:
            import ctranslate2  # noqa: PLC0415

            return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            return "cpu"
    try:
        import torch  # noqa: PLC0415

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="transcribe",  # argv[0] is the full .exe path once installed
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--version", action="version",
                   version=f"transcriber {__version__}")
    p.add_argument("audio", nargs="+", help="one or more audio/video files")
    p.add_argument("-m", "--model", default=None,
                   help=f"{', '.join(MODELS)}, or a Hugging Face id. "
                        f"Omit to be asked interactively (default: {DEFAULT_MODEL})")
    p.add_argument("-l", "--language", default=None,
                   help="ISO code, or 'auto' to detect. Omit to be asked "
                        "interactively (default: sv for kb-* models, else en)")
    p.add_argument("--task", default="transcribe", choices=["transcribe", "translate"])
    p.add_argument("--backend", default="auto",
                   choices=["auto", "faster-whisper", "transformers"])
    p.add_argument("--mode", default="longform", choices=["longform", "chunked"],
                   help="transformers only (default: longform)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--compute-type", default="auto",
                   choices=["auto", "int8", "int8_float16", "int8_bfloat16",
                            "float16", "bfloat16", "float32"],
                   help="faster-whisper precision (default: int8 on CPU, "
                        "float16 on GPU)")
    p.add_argument("--beams", type=int, default=5,
                   help="1 is fastest; 5 is more accurate (default: 5)")
    p.add_argument("--batch-size", type=int, default=0,
                   help="chunked mode; 0 = auto (1 on CPU, 8 on GPU)")
    p.add_argument("--chunk-s", type=float, default=30.0,
                   help=f"chunked mode window, max {WHISPER_WINDOW_S:.0f} "
                        "(default: 30)")
    p.add_argument("--overlap-s", type=float, default=5.0)
    p.add_argument("--vad", default=None, choices=["auto", "on", "off"],
                   help="drop non-speech before decoding. Omit to be asked "
                        f"interactively (default: {DEFAULT_VAD})")
    p.add_argument("--vocab", default="", metavar="TERMS",
                   help="comma-separated names/terms to bias toward, e.g. "
                        "\"EPAM, AstraZeneca\", or @path\\to\\file.txt. If omitted, "
                        "looks for <recording>.vocab.txt then vocab.txt beside "
                        "the audio")
    p.add_argument("--condition-on-previous", default="off", choices=["on", "off"],
                   help="feed earlier text back as context. Improves punctuation "
                        "and casing, but can make the model skip content on long "
                        "recordings (default: off)")
    p.add_argument("--vad-min-silence", type=int, default=400, metavar="MS",
                   help="silence this long or longer splits speech (default: 400)")
    p.add_argument("--formats", default="txt,json",
                   help="comma-separated: txt, srt, vtt, json "
                        "(default: txt,json - use --formats txt for text only)")
    p.add_argument("-o", "--output-dir", default=None,
                   help="default: alongside each input file")
    p.add_argument("--no-minute-markers", action="store_true",
                   help="omit [hh:mm:ss] markers from the .txt output")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    formats = [f.strip().lower() for f in args.formats.split(",") if f.strip()]
    unknown = [f for f in formats if f not in WRITERS]
    if unknown:
        raise SystemExit(f"Unknown output format(s): {', '.join(unknown)}")

    if args.chunk_s > WHISPER_WINDOW_S:
        raise SystemExit(
            f"--chunk-s cannot exceed {WHISPER_WINDOW_S:.0f}: Whisper's "
            "receptive field is fixed at that length and anything longer is "
            "silently truncated, losing the remainder of the window."
        )

    # An explicit --vocab is the same for every recording, so read it once and
    # now: a bad path should not wait for the model to load.
    explicit_vocab: List[str] = []
    if args.vocab:
        try:
            explicit_vocab = load_vocab(args.vocab)
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc

    model_name = args.model or prompt_for_model()
    backend_name = pick_backend(args.backend, model_name)
    model_id = resolve_model(model_name, backend_name)
    device = pick_device(args.device, backend_name)

    # Ask rather than silently assuming English: picking kb-large from the menu
    # and getting a translation back is a trap nobody walks into on purpose.
    lang_choice = args.language
    if lang_choice is None:
        lang_choice = prompt_for_language(default_language(model_name))
    language = None if lang_choice.lower() in ("auto", "none", "") else lang_choice

    if backend_name == "transformers" and args.mode == "chunked":
        batch_size = args.batch_size or (8 if device == "cuda" else 1)
    else:
        batch_size = max(1, args.batch_size)

    # Only ask when the setting has an effect: transformers long-form decoding
    # does no VAD at all.
    vad_choice = args.vad
    if vad_choice is None:
        if backend_name == "faster-whisper" or args.mode == "chunked":
            vad_choice = prompt_for_vad()
        else:
            vad_choice = DEFAULT_VAD

    # auto: on for faster-whisper (bundled, no extra install), off for
    # transformers chunked mode (needs the silero-vad package)
    if vad_choice == "auto":
        vad_on = backend_name == "faster-whisper"
    else:
        vad_on = vad_choice == "on"

    print(f"Backend: {backend_name}  model: {model_id}  device: {device}"
          f"  vad: {'on' if vad_on else 'off'}", file=sys.stderr)

    compute_type = None
    if backend_name == "faster-whisper":
        compute_type = args.compute_type
        if compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"
        backend = FasterWhisperBackend(
            model_id, device, compute_type, language, args.task, args.beams,
            vad=vad_on, vad_min_silence=args.vad_min_silence,
            condition_on_previous=args.condition_on_previous == "on",
        )
    else:
        backend = TransformersBackend(
            model_id, device, language, args.task, args.beams, batch_size,
            condition_on_previous=args.condition_on_previous == "on",
        )

    failures = 0
    warned_speed = False
    for path in args.audio:
        try:
            audio = load_audio(path)
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            failures += 1
            continue

        duration = len(audio) / SR
        print(f"\n{path}  ({format_timestamp(duration)})", file=sys.stderr)
        if (backend_name == "transformers" and device == "cpu"
                and duration > 600 and not warned_speed):
            print("  note: 'pip install faster-whisper' would run this several "
                  "times faster on CPU, with a progress bar", file=sys.stderr)
            warned_speed = True

        if args.vocab:
            vocab = explicit_vocab
            # only a file has a name worth printing; inline terms are the text
            vocab_source = args.vocab[1:] if args.vocab.startswith("@") else None
        else:
            found = find_vocab(path)
            try:
                vocab = load_vocab("@" + found) if found else []
            except RuntimeError as exc:
                # carrying on without it would quietly get the names wrong
                print(f"  error: {path}: {exc}", file=sys.stderr)
                failures += 1
                continue
            vocab_source = found
        if vocab:
            origin = (f" from {os.path.basename(vocab_source)}"
                      if vocab_source else "")
            print(f"  vocab: {len(vocab)} terms{origin}", file=sys.stderr)
            if backend_name != "faster-whisper":
                print("  note: vocabulary biasing needs the faster-whisper "
                      "backend; ignored here", file=sys.stderr)

        started = time.monotonic()
        try:
            if backend_name == "faster-whisper":
                segments = backend.transcribe(audio, vocab)
            elif args.mode == "longform":
                segments = backend.transcribe_longform(audio)
            else:
                if vad_on:
                    windows = plan_vad_chunks(audio, SR, args.chunk_s,
                                              min_silence_ms=args.vad_min_silence)
                    dedup = False  # silence boundaries mean nothing to de-duplicate
                else:
                    windows = plan_fixed_chunks(len(audio), SR, args.chunk_s,
                                                args.overlap_s)
                    dedup = True
                if not windows:
                    print("  no speech detected", file=sys.stderr)
                    segments = []
                else:
                    segments = backend.transcribe_chunked(audio, windows, dedup=dedup)
        except Exception as exc:  # noqa: BLE001 - one bad file must not end the batch
            print(f"  error: {path}: {exc}", file=sys.stderr)
            failures += 1
            continue

        elapsed = time.monotonic() - started
        if elapsed > 0:
            print(f"  took {elapsed / 60:.1f} min for {duration / 60:.1f} min of "
                  f"audio ({duration / elapsed:.1f}x realtime)", file=sys.stderr)

        if not segments:
            print("  produced no text", file=sys.stderr)

        out_dir = args.output_dir or os.path.dirname(os.path.abspath(path))
        try:
            os.makedirs(out_dir, exist_ok=True)
        except OSError as exc:
            print(f"  error: cannot create {out_dir}: {exc}", file=sys.stderr)
            failures += 1
            continue
        stem = os.path.splitext(os.path.basename(path))[0]
        meta = {
            "source": os.path.basename(path),
            "duration": round(duration, 2),
            "transcribed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "elapsed_s": round(elapsed, 1),
            "realtime_factor": round(duration / elapsed, 2) if elapsed > 0 else None,
            "settings": {
                "model": model_id,
                "backend": backend_name,
                "language": language or "auto",
                "task": args.task,
                "beams": args.beams,
                "vad": vad_on,
                "vad_min_silence_ms": args.vad_min_silence if vad_on else None,
                "condition_on_previous": args.condition_on_previous == "on",
                "device": device,
                "compute_type": compute_type,
                "mode": args.mode if backend_name == "transformers" else None,
                "vocab_terms": list(vocab),
                "vocab_source": (os.path.basename(vocab_source)
                                 if vocab and vocab_source else None),
            },
            "script_version": SCRIPT_VERSION,
        }
        try:
            for fmt in formats:
                out_path = os.path.join(out_dir, f"{stem}.{fmt}")
                if fmt == "txt":
                    write_txt(segments, out_path,
                              minute_markers=not args.no_minute_markers)
                elif fmt == "json":
                    write_json(segments, out_path, meta)
                else:
                    WRITERS[fmt](segments, out_path)
                print(f"  wrote {out_path}", file=sys.stderr)
        except OSError as exc:  # full disk, read-only dir, locked file
            print(f"  error: could not write output for {path}: {exc}",
                  file=sys.stderr)
            failures += 1

    return 1 if failures else 0


def run() -> None:
    """Console-script entry point: main() plus Ctrl-C handling."""
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    run()