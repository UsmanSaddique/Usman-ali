"""
AI Director — TTS Service
Primary engine: Kokoro-82M (local, Apache-2.0) — production-quality narration.
Fallback engine: Meta MMS-TTS (per-language VITS) for non-English voices.
Word timestamps + transcribe-back QA: faster-whisper.
Legacy WhisperX path kept for the old song-mode flow.
"""
import os
import time
import json
import logging
import subprocess
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)


@dataclass
class WordTimestamp:
    word: str
    start: float   # seconds
    end: float     # seconds


@dataclass
class TTSResult:
    audio_path: str
    duration: float
    sample_rate: int
    text: str
    word_timestamps: list[WordTimestamp] = field(default_factory=list)
    generation_time: float = 0.0


@dataclass
class NarrationSegment:
    scene_number: int
    text: str
    audio_path: str
    start_time: float   # offset in the full narration track
    end_time: float
    word_timestamps: list[WordTimestamp] = field(default_factory=list)


@dataclass
class BeatAudio:
    """One narration beat's audio inside the master WAV (narration mode)."""
    beat_index: int          # global beat index (1-based, across chapters)
    chapter_index: int       # 1-based
    text: str
    audio_path: str          # per-beat WAV
    start: float             # EXACT offset in the master WAV (sample-accurate)
    end: float
    wer: float = 0.0         # transcribe-back word error rate (QA)


@dataclass
class NarrationMaster:
    """The master narration track — the TIMELINE MASTER for narration mode."""
    audio_path: str                       # loudness-normalized master WAV
    duration: float
    sample_rate: int
    beats: list = field(default_factory=list)   # list[BeatAudio]
    words: list = field(default_factory=list)   # list[WordTimestamp], absolute times
    timing_json: str = ""                 # sidecar with beats+words (for SRT/QA)


class TTSService:
    """
    Generate narration audio using WanGP Omnivoice local API.
    Then run WhisperX forced alignment for word-level timestamps.
    """

    # Language -> Meta MMS-TTS model. Fully local, ~150MB each, runs on CPU.
    # Meta MMS-TTS has no standalone Urdu voice, but Hindi exists and spoken
    # Hindi≈Urdu (Hindustani) — so "urdu" routes to the Hindi voice (needs
    # Devanagari-script input to sound right). Punjabi also available.
    MMS_MODELS = {
        "english": "facebook/mms-tts-eng",
        "en": "facebook/mms-tts-eng",
        "urdu": "facebook/mms-tts-hin",   # Hindustani voice (sounds like spoken Urdu)
        "ur": "facebook/mms-tts-hin",
        "hindi": "facebook/mms-tts-hin",
        "hi": "facebook/mms-tts-hin",
        "punjabi": "facebook/mms-tts-pan",
        # Roman Urdu has no native model — read with the English voice
        # (Latin-script phonemes); approximate but intelligible.
        "roman urdu": "facebook/mms-tts-eng",
    }

    def __init__(self, config):
        self.config = config
        self.default_speed = getattr(config.tts, "default_speed", 0.95)
        self.default_language = "english"
        self._loaded = {}  # model_id -> (model, tokenizer)
        self._kokoro = {}  # lang_code -> KPipeline
        self._whisper = None  # faster-whisper model (lazy)

    # ── Kokoro engine (primary, English) ───────────────────────────────

    KOKORO_SR = 24000

    def _kokoro_pipe(self, voice: str):
        """Lazy-load a Kokoro KPipeline for the voice's language.
        Voice prefixes: a* = American EN, b* = British EN."""
        lang_code = (voice or "a")[0].lower()
        if lang_code not in ("a", "b"):
            lang_code = "a"
        if lang_code not in self._kokoro:
            from kokoro import KPipeline
            repo = getattr(self.config.tts, "kokoro_repo", "hexgrad/Kokoro-82M")
            logger.info(f"[TTS] Loading Kokoro-82M pipeline (lang={lang_code})")
            self._kokoro[lang_code] = KPipeline(lang_code=lang_code, repo_id=repo)
        return self._kokoro[lang_code]

    def kokoro_generate(self, text: str, voice: Optional[str] = None,
                        speed: Optional[float] = None):
        """Text → float32 waveform @24kHz via Kokoro. Returns numpy array."""
        import numpy as np
        voice = voice or getattr(self.config.tts, "kokoro_voice", "am_michael")
        speed = float(speed or getattr(self.config.tts, "kokoro_speed", 1.0))
        pipe = self._kokoro_pipe(voice)
        chunks = []
        for _gs, _ps, audio in pipe(text, voice=voice, speed=speed):
            a = audio.detach().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio)
            chunks.append(a.astype(np.float32).squeeze())
        if not chunks:
            raise RuntimeError(f"Kokoro produced no audio for: {text[:60]!r}")
        return np.concatenate(chunks)

    def unload_engines(self):
        """Free TTS + whisper models (they're small, but VRAM discipline)."""
        self._kokoro.clear()
        self._whisper = None
        self._loaded.clear()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    # ── faster-whisper (word timestamps + transcribe-back QA) ──────────

    def _whisper_model(self):
        if self._whisper is None:
            from faster_whisper import WhisperModel
            name = getattr(self.config.tts, "whisper_model", "small")
            try:
                self._whisper = WhisperModel(name, device="cuda", compute_type="float16")
                logger.info(f"[TTS] faster-whisper '{name}' on CUDA")
            except Exception as e:
                logger.info(f"[TTS] faster-whisper CUDA unavailable ({e}); using CPU int8")
                self._whisper = WhisperModel(name, device="cpu", compute_type="int8")
        return self._whisper

    def transcribe_words(self, audio_path: str, language: str = "en"):
        """Transcribe → (full_text, list[WordTimestamp])."""
        model = self._whisper_model()
        segments, _info = model.transcribe(
            audio_path, language=language, word_timestamps=True,
            vad_filter=False, beam_size=5)
        words, texts = [], []
        for seg in segments:
            texts.append(seg.text)
            for w in (seg.words or []):
                words.append(WordTimestamp(word=w.word.strip(),
                                           start=float(w.start),
                                           end=float(w.end)))
        return " ".join(t.strip() for t in texts).strip(), words

    @staticmethod
    def _wer(ref: str, hyp: str) -> float:
        """Word error rate via edit distance on normalized words."""
        import re as _re

        def norm(s):
            return _re.sub(r"[^a-z0-9' ]+", " ", s.lower()).split()
        r, h = norm(ref), norm(hyp)
        if not r:
            return 0.0
        d = list(range(len(h) + 1))
        for i in range(1, len(r) + 1):
            prev, d[0] = d[0], i
            for j in range(1, len(h) + 1):
                cur = d[j]
                d[j] = min(d[j] + 1, d[j - 1] + 1,
                           prev + (0 if r[i - 1] == h[j - 1] else 1))
                prev = cur
        return d[len(h)] / len(r)

    # ── Narration-mode master track ────────────────────────────────────

    @staticmethod
    def _normalize_speech_text(text: str) -> str:
        """Strip characters that make TTS stumble (markdown, odd symbols)."""
        import re as _re
        t = _re.sub(r"[*_#`~^\[\]{}<>|\\]", " ", text)
        t = _re.sub(r"\s+", " ", t)
        return t.strip()

    # ── Chatterbox engine (Voice Cloning) ────────────────────────────────────

    def chatterbox_generate(self, text: str, voice_ref: str, exaggeration: float = 1.0):
        """Text → float32 waveform @24kHz via ResembleAI/chatterbox.
        Clones the voice from `voice_ref`."""
        import numpy as np
        from chatterbox import ChatterBox
        import librosa
        import torchaudio

        if not hasattr(self, "_chatterbox_model"):
            logger.info("[TTS] Loading Chatterbox model")
            self._chatterbox_model = ChatterBox.from_pretrained("ResembleAI/chatterbox")
            if hasattr(self._chatterbox_model, "to"):
                self._chatterbox_model = self._chatterbox_model.to("cuda")

        # Load reference audio
        ref_wav, sr = librosa.load(voice_ref, sr=24000)
        
        # Chatterbox generation
        # Chatterbox outputs at 24kHz (usually).
        out_wav = self._chatterbox_model.generate(
            text=text,
            voice_prompt=ref_wav,
            exaggeration=exaggeration,
        )
        a = out_wav.detach().cpu().numpy() if hasattr(out_wav, "detach") else np.asarray(out_wav)
        return a.astype(np.float32).squeeze()

    def generate_narration_master(
        self,
        beats: list[dict],
        output_dir: str,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        voice_ref: Optional[str] = None,
    ) -> NarrationMaster:
        """
        Narration mode: per-beat Kokoro/Chatterbox WAVs → sample-accurate master concat
        with breath pauses → loudness-normalized master → word timestamps +
        per-beat transcribe-back WER QA.

        `beats`: [{"text": ..., "chapter_index": 1-based, "chapter_break": bool}]
        Beat start/end offsets are EXACT (computed from sample counts, not
        alignment) — they are the timeline contract for the whole pipeline.
        """
        import json as _json
        import numpy as np
        import scipy.io.wavfile

        sr = self.KOKORO_SR
        out_dir = Path(output_dir) / "narration"
        out_dir.mkdir(parents=True, exist_ok=True)
        pause_beat = float(getattr(self.config.tts, "pause_beat", 0.35))
        pause_chapter = float(getattr(self.config.tts, "pause_chapter", 0.9))
        wer_flag = float(getattr(self.config.tts, "wer_flag_threshold", 0.12))

        t0 = time.time()
        beat_audios: list[BeatAudio] = []
        waves: list = []
        cursor = 0  # samples
        
        use_chatterbox = voice == "chatterbox" and voice_ref and Path(voice_ref).exists()
        if use_chatterbox:
            logger.info(f"[TTS] Using Chatterbox voice cloning with ref: {voice_ref}")

        for i, beat in enumerate(beats):
            text = self._normalize_speech_text(str(beat.get("text", "")))
            if not text:
                continue
            gidx = i + 1
            
            try:
                if use_chatterbox:
                    wav = self.chatterbox_generate(text, voice_ref=voice_ref)
                else:
                    wav = self.kokoro_generate(text, voice=voice, speed=speed)
            except Exception as e:
                logger.warning(f"[TTS] TTS failed for beat {gidx}: {e}")
                wav = np.zeros(int(sr * 1.0), dtype=np.float32)

            beat_path = str(out_dir / f"beat_{gidx:03d}.wav")
            scipy.io.wavfile.write(beat_path, sr, (wav * 32767).astype(np.int16))

            start = cursor / sr
            end = (cursor + len(wav)) / sr
            beat_audios.append(BeatAudio(
                beat_index=gidx,
                chapter_index=int(beat.get("chapter_index", 1)),
                text=text, audio_path=beat_path, start=start, end=end))
            waves.append(wav)
            cursor += len(wav)

            # breath pause after the beat (longer at chapter breaks)
            pause = pause_chapter if beat.get("chapter_break") else pause_beat
            if i < len(beats) - 1 and pause > 0:
                gap = np.zeros(int(pause * sr), dtype=np.float32)
                waves.append(gap)
                cursor += len(gap)

        if not beat_audios:
            raise ValueError("No narration beats with text")

        master = np.concatenate(waves)
        raw_path = str(out_dir / "narration_master_raw.wav")
        scipy.io.wavfile.write(raw_path, sr, (master * 32767).astype(np.int16))
        logger.info(f"[TTS] Kokoro narration: {len(beat_audios)} beats, "
                    f"{cursor / sr:.1f}s in {time.time() - t0:.1f}s")

        # Loudness-normalize the voice bus (single pass keeps duration)
        master_path = str(out_dir / "narration_master.wav")
        lufs = float(getattr(self.config.tts, "narration_lufs", -16.0))
        try:
            subprocess.run([
                self.config.paths.ffmpeg_bin, "-y", "-i", raw_path,
                "-af", f"loudnorm=I={lufs}:TP=-1.5:LRA=11",
                "-ar", str(sr), "-c:a", "pcm_s16le", master_path,
            ], capture_output=True, check=True, timeout=300)
        except Exception as e:
            logger.warning(f"[TTS] loudnorm failed, using raw master: {e}")
            master_path = raw_path

        # QA: transcribe-back — word timestamps on the master + per-beat WER
        words: list[WordTimestamp] = []
        try:
            _text, words = self.transcribe_words(master_path)
            for ba in beat_audios:
                hyp, _ = self.transcribe_words(ba.audio_path)
                ba.wer = round(self._wer(ba.text, hyp), 3)
                if ba.wer > wer_flag:
                    logger.warning(f"[TTS] Beat {ba.beat_index} WER {ba.wer:.0%}: "
                                   f"'{ba.text[:60]}...' heard as '{hyp[:60]}...'")
        except Exception as e:
            logger.warning(f"[TTS] transcribe-back QA failed (non-fatal): {e}")

        duration = self._get_audio_duration(master_path)
        timing_json = str(out_dir / "narration_timing.json")
        with open(timing_json, "w", encoding="utf-8") as f:
            _json.dump({
                "audio_path": master_path,
                "duration": duration,
                "sample_rate": sr,
                "voice": voice or getattr(self.config.tts, "kokoro_voice", ""),
                "beats": [{
                    "beat_index": b.beat_index, "chapter_index": b.chapter_index,
                    "text": b.text, "start": round(b.start, 3),
                    "end": round(b.end, 3), "wer": b.wer,
                } for b in beat_audios],
                "words": [{"word": w.word, "start": round(w.start, 3),
                           "end": round(w.end, 3)} for w in words],
            }, f, ensure_ascii=False, indent=1)

        return NarrationMaster(
            audio_path=master_path, duration=duration, sample_rate=sr,
            beats=beat_audios, words=words, timing_json=timing_json)

    def _get_model(self, language: str):
        """Lazy-load and cache an MMS-TTS model for a language."""
        model_id = self.MMS_MODELS.get(str(language).lower().strip(),
                                       "facebook/mms-tts-eng")
        if model_id not in self._loaded:
            from transformers import VitsModel, AutoTokenizer
            logger.info(f"[TTS] Loading MMS-TTS model '{model_id}' (lang={language})")
            model = VitsModel.from_pretrained(model_id)
            tok = AutoTokenizer.from_pretrained(model_id)
            model.eval()
            self._loaded[model_id] = (model, tok)
        return self._loaded[model_id]

    # ── Single Text → Speech ───────────────────────────────────────────

    def generate(
        self,
        text: str,
        output_path: str,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        language: Optional[str] = None,
    ) -> TTSResult:
        """Generate speech for one text. English routes to Kokoro (production
        quality) when engine=kokoro; other languages use Meta MMS-TTS."""
        import torch
        import numpy as np
        import scipy.io.wavfile

        language = language or self.default_language
        speed = speed or self.default_speed
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        engine = getattr(self.config.tts, "engine", "kokoro")
        if engine == "kokoro" and str(language).lower().strip() in (
                "english", "en", "roman urdu"):
            try:
                t0 = time.time()
                wav = self.kokoro_generate(text, voice=voice)
                scipy.io.wavfile.write(output_path, self.KOKORO_SR,
                                       (wav * 32767).astype(np.int16))
                dur = len(wav) / self.KOKORO_SR
                logger.info(f"[TTS] Kokoro: {dur:.1f}s audio in {time.time()-t0:.1f}s")
                return TTSResult(audio_path=output_path, duration=dur,
                                 sample_rate=self.KOKORO_SR, text=text,
                                 generation_time=time.time() - t0)
            except Exception as e:
                logger.warning(f"[TTS] Kokoro failed ({e}); falling back to MMS")

        t0 = time.time()
        logger.info(f"[TTS] Generating ({language}): '{text[:60]}...'")

        model, tok = self._get_model(language)
        # speaking_rate < 1.0 slows speech (gentle pacing for kids content)
        try:
            model.speaking_rate = float(speed)
        except Exception:
            pass

        inputs = tok(text, return_tensors="pt")
        with torch.no_grad():
            waveform = model(**inputs).waveform  # [1, samples]
        wav = waveform.squeeze().cpu().numpy()
        sr = int(model.config.sampling_rate)
        scipy.io.wavfile.write(output_path, sr, (wav * 32767).astype(np.int16))

        elapsed = time.time() - t0
        duration = len(wav) / sr
        logger.info(f"[TTS] Generated {duration:.1f}s audio in {elapsed:.1f}s ({sr}Hz)")

        return TTSResult(
            audio_path=output_path,
            duration=duration,
            sample_rate=sr,
            text=text,
            generation_time=elapsed,
        )

    # ── Full Narration (all scenes) ────────────────────────────────────

    def generate_full_narration(
        self,
        scenes: list[dict],
        output_dir: str,
        voice: Optional[str] = None,
        pause_between: float = 0.5,
        language: Optional[str] = None,
    ) -> tuple[str, list[NarrationSegment]]:
        """
        Generate narration for all scenes that have narration_text.
        Returns: (combined_audio_path, list of NarrationSegments)

        Each scene gets its own audio file, then they're concatenated
        with silence gaps matching scene timing.
        """
        segments: list[NarrationSegment] = []
        audio_files: list[tuple[str, float]] = []  # (path, pause_after)
        out_dir = Path(output_dir) / "narration"
        out_dir.mkdir(parents=True, exist_ok=True)

        for scene in scenes:
            text = scene.get("narration_text", "").strip()
            if not text:
                continue

            scene_num = scene["scene_number"]
            audio_path = str(out_dir / f"narration_{scene_num:03d}.wav")

            result = self.generate(text, audio_path, voice=voice, language=language)

            segment = NarrationSegment(
                scene_number=scene_num,
                text=text,
                audio_path=audio_path,
                start_time=0.0,  # calculated after concat
                end_time=result.duration,
            )
            segments.append(segment)
            audio_files.append((audio_path, pause_between))

        if not audio_files:
            logger.info("[TTS] No narration text in any scene")
            return "", []

        # Concatenate all narration segments with pauses
        combined_path = str(out_dir / "narration_full.wav")
        self._concat_audio_with_pauses(audio_files, combined_path)

        # Run WhisperX alignment on the combined audio
        try:
            timestamps = self.align_audio(combined_path, " ".join(
                s.text for s in segments
            ))
            # Distribute timestamps back to segments
            self._distribute_timestamps(segments, timestamps)
        except Exception as e:
            logger.warning(f"[TTS] WhisperX alignment failed: {e}")

        # Calculate running time offsets
        running_time = 0.0
        for seg in segments:
            seg.start_time = running_time
            seg_duration = self._get_audio_duration(seg.audio_path)
            seg.end_time = running_time + seg_duration
            running_time = seg.end_time + pause_between

        return combined_path, segments

    # ── WhisperX Forced Alignment ──────────────────────────────────────

    def align_audio(
        self,
        audio_path: str,
        transcript: str,
    ) -> list[WordTimestamp]:
        """
        Run WhisperX forced alignment to get word-level timestamps.
        Returns list of WordTimestamp for each word.
        """
        logger.info(f"[TTS] Running WhisperX alignment on {audio_path}")

        try:
            import whisperx

            # Load alignment model (uses GPU if available, but small footprint)
            device = "cuda"
            model = whisperx.load_align_model(language_code="en", device=device)

            # WhisperX needs segments in a specific format
            audio = whisperx.load_audio(audio_path)

            # Create a "transcript" format WhisperX expects
            segments = [{
                "text": transcript,
                "start": 0.0,
                "end": self._get_audio_duration(audio_path),
            }]

            result = whisperx.align(
                segments, model, {"waveform": audio, "sample_rate": 16000},
                device=device,
                return_char_alignments=False,
            )

            timestamps = []
            for seg in result.get("segments", []):
                for word_info in seg.get("words", []):
                    timestamps.append(WordTimestamp(
                        word=word_info.get("word", ""),
                        start=word_info.get("start", 0.0),
                        end=word_info.get("end", 0.0),
                    ))

            logger.info(f"[TTS] Aligned {len(timestamps)} words")
            return timestamps

        except ImportError:
            logger.warning("[TTS] whisperx not installed, skipping alignment")
            return []

    # ── Audio Utilities ────────────────────────────────────────────────

    def _concat_audio_with_pauses(
        self,
        audio_files: list[tuple[str, float]],  # (path, pause_seconds_after)
        output_path: str,
    ):
        """Concatenate audio files with silence gaps using FFmpeg."""
        ffmpeg = self.config.paths.ffmpeg_bin

        # Build complex filter with silence pads
        inputs = []
        filter_parts = []
        concat_inputs = []

        for i, (path, pause) in enumerate(audio_files):
            inputs.extend(["-i", path])
            # Add the audio segment
            concat_inputs.append(f"[{i}:a]")
            # Add silence if not last
            if i < len(audio_files) - 1 and pause > 0:
                sr = 22050  # sample rate
                silence_id = f"s{i}"
                filter_parts.append(
                    f"aevalsrc=0:d={pause}:s={sr}:c=mono[{silence_id}]"
                )
                concat_inputs.append(f"[{silence_id}]")

        n_inputs = len(concat_inputs)
        concat_str = "".join(concat_inputs)
        filter_parts.append(f"{concat_str}concat=n={n_inputs}:v=0:a=1[out]")

        filter_graph = ";".join(filter_parts)

        cmd = [
            ffmpeg, "-y",
            *inputs,
            "-filter_complex", filter_graph,
            "-map", "[out]",
            "-c:a", "pcm_s16le",
            output_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            # Fallback: simple concat without pauses
            self._simple_concat_audio(
                [f for f, _ in audio_files], output_path
            )

    def _simple_concat_audio(self, audio_files: list[str], output_path: str):
        """Fallback: simple concatenation via ffmpeg concat demuxer."""
        import tempfile
        ffmpeg = self.config.paths.ffmpeg_bin

        list_file = tempfile.mktemp(suffix=".txt")
        with open(list_file, 'w') as f:
            for path in audio_files:
                f.write(f"file '{path}'\n")

        cmd = [
            ffmpeg, "-y",
            "-f", "concat", "-safe", "0",
            "-i", list_file,
            "-c", "copy",
            output_path,
        ]
        subprocess.run(cmd, capture_output=True, check=True, timeout=120)
        os.remove(list_file)

    def _distribute_timestamps(
        self,
        segments: list[NarrationSegment],
        timestamps: list[WordTimestamp],
    ):
        """Distribute word timestamps back to their respective segments."""
        word_idx = 0
        for seg in segments:
            seg_words = seg.text.split()
            seg.word_timestamps = []
            for _ in seg_words:
                if word_idx < len(timestamps):
                    seg.word_timestamps.append(timestamps[word_idx])
                    word_idx += 1

    @staticmethod
    def _get_audio_duration(audio_path: str) -> float:
        """Duration in seconds. Reads the WAV header directly (no ffprobe needed,
        since MMS-TTS writes WAV). Falls back to 0.0 for non-WAV/unreadable files."""
        try:
            import wave
            with wave.open(audio_path, "rb") as w:
                frames = w.getnframes()
                rate = w.getframerate()
                if rate:
                    return frames / float(rate)
        except Exception:
            pass
        return 0.0
