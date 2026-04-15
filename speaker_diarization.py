#!/usr/bin/env python3

import os
import warnings
import numpy as np
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import threading
import time
import shutil
import torch
import soundfile as sf
from pyannote.audio import Pipeline
from faster_whisper import WhisperModel
try:
    from speechbrain.inference.speaker import EncoderClassifier
    from speechbrain.utils import fetching as sb_fetching
    from speechbrain.inference import interfaces as sb_interfaces
except ImportError:
    from speechbrain.pretrained import EncoderClassifier
    from speechbrain.pretrained import fetching as sb_fetching
    from speechbrain.pretrained import interfaces as sb_interfaces
from huggingface_hub import snapshot_download
from threshold_settings import ThresholdSettings
from translation_module import TranslationModule

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class DiarizationPipeline:
    def __init__(self, hf_token: str, transcription_lang: str = "yue"):
        """Initialize the diarization pipeline with performance improvements.
        
        Args:
            hf_token: HuggingFace API token
            transcription_lang: Language code for transcription (default: 'zh' for Chinese)
        """
        print("[DIARIZATION] Initializing Diarization Pipeline...")
        
        os.environ["HF_TOKEN"] = hf_token
        self.thresholds = ThresholdSettings.from_env()
        
        # Store default transcription language
        self.transcription_lang = transcription_lang
        normalized_lang = (transcription_lang or "").lower()
        self.prioritize_cantonese_english = normalized_lang in {
            "yue-en", "yue_en", "yue+en",
            "cantonese-en", "cantonese_en", "cantonese+en",
            "en-yue", "en_yue", "en+yue",
        }
        self.allow_english = self.prioritize_cantonese_english or normalized_lang in {"en", "english"}
        self.whisper_language = self._normalize_whisper_language(transcription_lang)
        self.cantonese_mode = self.prioritize_cantonese_english or normalized_lang in {"yue", "zh-hk", "zh-yue", "cantonese"}
        self.cantonese_initial_prompt = None
        self.prompt_echo_blocklist = [
            "以下內容以廣東話與英文為主",
            "請優先使用繁體中文轉寫廣東話",
            "英文內容保留英文原文",
            "不要轉成普通話書面語",
            "以下是廣東話對話",
            "請使用繁體中文準確轉寫口語內容",
        ]
        
        print(f"[DIARIZATION] Default transcription language: {self.transcription_lang}")
        print("[DIARIZATION] Translation language will be set per request")
        
        self.diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )
        
        self.speaker_sim_threshold = self.thresholds.speaker_sim_threshold
        self.wearer_sim_threshold = self.thresholds.wearer_sim_threshold
        self.diar_clustering_threshold = self.thresholds.diar_clustering_threshold
        self.diar_min_cluster_size = self.thresholds.diar_min_cluster_size
        self.diar_min_duration_off = self.thresholds.diar_min_duration_off

        # Using correct parameter names for pyannote/speaker-diarization-3.1
        try:
            self.diarization_pipeline.instantiate({
                "clustering": {
                    "threshold": self.diar_clustering_threshold,
                    "min_cluster_size": self.diar_min_cluster_size,
                },
                "segmentation": {
                    "min_duration_off": self.diar_min_duration_off,
                }
            })
            print("[DIARIZATION] Diarization configured for stricter speaker separation")
        except Exception as e:
            print(f"[DIARIZATION] Warning: Could not configure diarization parameters: {e}")
            print("[DIARIZATION] Using default diarization settings")
        
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        whisper_compute_type = "float16" if device_str == "cuda" else "int8_float32"

        self.chunk_transcribe_options = {
            "beam_size": 2,
            "temperature": 0,
            "vad_filter": True,
            "condition_on_previous_text": False,
            "word_timestamps": False,
        }
        self.full_transcribe_options = {
            "beam_size": 5,
            "temperature": 0,
            "vad_filter": True,
            "condition_on_previous_text": True,
        }
        self.speaker_match_threshold = self.thresholds.speaker_match_threshold
        self.speaker_merge_band_low = self.thresholds.speaker_merge_band_low
        self.speaker_merge_band_high = self.thresholds.speaker_merge_band_high
        self.speaker_proactive_merge_threshold = self.thresholds.speaker_proactive_merge_threshold
        self.chunk_speaker_match_threshold = self.speaker_match_threshold
        self.chunk_speaker_soft_match_threshold = self.speaker_merge_band_low
        self.tracked_speaker_merge_threshold = self.speaker_proactive_merge_threshold
        self.speaker_embedding_update_alpha = 0.10
        self.chunk_hold_speaker_threshold = self.speaker_merge_band_low
        self.chunk_min_rms_energy = 0.003
        self.chunk_min_peak_amplitude = 0.015
        self.chunk_min_avg_logprob = -1.15
        self.chunk_max_no_speech_prob = 0.60
        self.chunk_new_speaker_confirmations = 2
        self.chunk_pending_speaker_similarity = self.speaker_proactive_merge_threshold
        self.chunk_min_speech_duration = 0.35
        self.chunk_min_speech_fraction = 0.12
        self.chunk_min_speech_duration_for_switch = 0.85
        self.chunk_min_speech_fraction_for_switch = 0.30
        self.chunk_recent_speaker_window_sec = 6.0
        self.chunk_speaker_switch_margin = 0.08
        self.chunk_embedding_min_speech_duration = 0.45
        self.registration_segment_duration = 1.6
        self.registration_overlap_duration = 0.5
        self.registration_min_segment_duration = 1.0
        self.registration_min_rms_energy = 0.003
        self.registration_centroid_min_similarity = 0.58
        self.registration_pairwise_min_similarity = 0.55
        self.wearer_multi_vote_sim_threshold = self.thresholds.wearer_multi_vote_sim_threshold
        self.wearer_chunk_avg_threshold = self.thresholds.wearer_chunk_avg_threshold
        self.wearer_chunk_median_threshold = self.thresholds.wearer_chunk_median_threshold
        self.wearer_chunk_max_threshold = self.thresholds.wearer_chunk_max_threshold
        self.wearer_chunk_match_ratio_threshold = self.thresholds.wearer_chunk_match_ratio_threshold
        self.wearer_chunk_match_ratio_max_threshold = self.thresholds.wearer_chunk_match_ratio_max_threshold
        self.wearer_chunk_count_sim_threshold = self.thresholds.wearer_chunk_count_sim_threshold
        self.wearer_chunk_count_ratio_threshold = self.thresholds.wearer_chunk_count_ratio_threshold
        self.wearer_single_threshold_chunk = self.thresholds.wearer_single_threshold_chunk
        self.wearer_full_avg_threshold = self.thresholds.wearer_full_avg_threshold
        self.wearer_full_median_threshold = self.thresholds.wearer_full_median_threshold
        self.wearer_full_median_max_threshold = self.thresholds.wearer_full_median_max_threshold
        self.wearer_full_match_ratio_threshold = self.thresholds.wearer_full_match_ratio_threshold
        self.wearer_full_match_ratio_max_threshold = self.thresholds.wearer_full_match_ratio_max_threshold
        self.wearer_full_count_sim_threshold = self.thresholds.wearer_full_count_sim_threshold
        self.wearer_full_count_ratio_threshold = self.thresholds.wearer_full_count_ratio_threshold
        self.wearer_single_threshold_full = self.thresholds.wearer_single_threshold_full
        self.wearer_recent_single_match_threshold = self.thresholds.wearer_sim_threshold

        self.whisper_model = WhisperModel(
            "large-v3",
            device=device_str,
            compute_type=whisper_compute_type,
            num_workers=1,  # Limit workers to save memory
            download_root=None,  # Use default cache
            local_files_only=False
        )
        print(f"[DIARIZATION] Whisper compute type: {whisper_compute_type}")
        print(f"[DIARIZATION] Chunk transcription options: {self.chunk_transcribe_options}")
        print(f"[DIARIZATION] Full transcription options: {self.full_transcribe_options}")
        print(
            f"[DIARIZATION] Speaker tracking thresholds: "
            f"match={self.chunk_speaker_match_threshold}, "
            f"soft_match={self.chunk_speaker_soft_match_threshold}, "
            f"merge={self.tracked_speaker_merge_threshold}"
        )
        if self.cantonese_mode:
            print(f"[DIARIZATION] Cantonese-focused mode enabled (Whisper language hint: {self.whisper_language})")
        
        # Initialize speaker embedding model (ECAPA-TDNN for 192-dim embeddings)
        print("[DIARIZATION] Loading speaker embedding model...")
        speaker_model_source = "speechbrain/spkrec-ecapa-voxceleb"
        speaker_model_dir = Path("pretrained_models/spkrec-ecapa-voxceleb")
        if os.name == "nt":
            if not getattr(Path, "_factory_safe_symlink_patch", False):
                original_symlink_to = Path.symlink_to

                def safe_symlink_to(path_obj, target, target_is_directory=False):
                    try:
                        return original_symlink_to(path_obj, target, target_is_directory=target_is_directory)
                    except OSError as e:
                        if getattr(e, "winerror", None) != 1314:
                            raise

                        target_path = Path(target)
                        path_obj.parent.mkdir(parents=True, exist_ok=True)
                        if path_obj.exists() or path_obj.is_symlink():
                            if path_obj.is_dir() and not path_obj.is_symlink():
                                shutil.rmtree(path_obj)
                            else:
                                path_obj.unlink()
                        if target_path.is_dir():
                            shutil.copytree(target_path, path_obj, dirs_exist_ok=True)
                        else:
                            shutil.copy2(target_path, path_obj)

                Path.symlink_to = safe_symlink_to
                Path._factory_safe_symlink_patch = True

            print("[DIARIZATION] Pre-downloading SpeechBrain model for Windows compatibility...")
            snapshot_download(
                repo_id=speaker_model_source,
                local_dir=str(speaker_model_dir),
                local_dir_use_symlinks=False
            )
            speaker_model_source = str(speaker_model_dir.resolve())

            original_fetch = sb_fetching.fetch

            def windows_safe_fetch(filename, source, savedir="./pretrained_model_checkpoints",
                                   overwrite=False, save_filename=None, use_auth_token=False,
                                   revision=None, cache_dir=None, silent_local_fetch=False):
                save_name = save_filename or filename
                savedir_path = Path(savedir)
                savedir_path.mkdir(parents=True, exist_ok=True)
                destination = savedir_path / save_name

                fetch_from = None
                if isinstance(source, sb_fetching.FetchSource):
                    fetch_from, source = source

                source_dir = Path(source)
                if source_dir.is_dir() and fetch_from not in [
                    sb_fetching.FetchFrom.HUGGING_FACE,
                    sb_fetching.FetchFrom.URI,
                ]:
                    sourcepath = (source_dir / filename).resolve()
                    if not sourcepath.exists():
                        raise ValueError(f"File not found: {sourcepath}")
                    if destination.exists() and not overwrite:
                        return destination
                    if destination.resolve() == sourcepath:
                        return destination
                    shutil.copy2(sourcepath, destination)
                    return destination

                try:
                    return original_fetch(
                        filename=filename,
                        source=source,
                        savedir=savedir,
                        overwrite=overwrite,
                        save_filename=save_filename,
                        use_auth_token=use_auth_token,
                        revision=revision,
                        cache_dir=cache_dir,
                        silent_local_fetch=silent_local_fetch,
                    )
                except OSError as e:
                    if getattr(e, "winerror", None) != 1314:
                        raise
                    raise

            sb_fetching.fetch = windows_safe_fetch
            sb_interfaces.fetch = windows_safe_fetch

        self.speaker_model = EncoderClassifier.from_hparams(
            source=speaker_model_source,
            savedir=str(speaker_model_dir),
            run_opts={"device": device_str}
        )
        print("[DIARIZATION] Speaker embedding model loaded (192-dim ECAPA-TDNN)")
        
        # Initialize translation module (always available for dynamic use)
        self.translator = None
        try:
            print("[DIARIZATION] Initializing translation module...")
            self.translator = TranslationModule(device=device_str)
            print("[DIARIZATION] Translation module initialized and ready")
        except Exception as e:
            print(f"[DIARIZATION] Warning: Could not initialize translation module: {e}")
            print("[DIARIZATION] Translation will be disabled")
            self.translator = None
        
        # Minimal cache for memory efficiency
        self.audio_cache = {}
        self.cache_lock = threading.Lock()
        self.max_cache_size = 5  # Limit cache size
        
        # Memory management
        self._cleanup_memory()
        
        print("[DIARIZATION] Optimized Diarization Pipeline initialized successfully")
        print(f"[DIARIZATION] Using device: {device_str}")

    def _normalize_whisper_language(self, language_code: str) -> Optional[str]:
        if not language_code:
            return "zh"

        normalized = language_code.lower()
        if normalized in {
            "yue-en", "yue_en", "yue+en",
            "cantonese-en", "cantonese_en", "cantonese+en",
            "en-yue", "en_yue", "en+yue",
        }:
            return None
        if normalized in {"yue", "zh-hk", "zh-yue", "cantonese"}:
            return "zh"
        if normalized == "english":
            return "en"
        return normalized

    def _resolve_source_language(self, requested_lang: str, info) -> str:
        normalized = (requested_lang or self.transcription_lang or "").lower()
        detected_lang = (getattr(info, "language", "") or "").lower()

        if normalized in {
            "yue-en", "yue_en", "yue+en",
            "cantonese-en", "cantonese_en", "cantonese+en",
            "en-yue", "en_yue", "en+yue",
        }:
            if detected_lang == "en":
                return "en"
            if detected_lang == "zh":
                return "yue"
            return "yue"

        if normalized in {"yue", "zh-hk", "zh-yue", "cantonese"}:
            return "yue"

        if normalized == "english":
            return "en"

        return requested_lang or self.transcription_lang

    def _strip_prompt_echo(self, text: str) -> str:
        if not text:
            return text

        cleaned = text.strip()
        for phrase in self.prompt_echo_blocklist:
            cleaned = cleaned.replace(phrase, " ")

        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ，。,.")
        return cleaned

    def _compute_rms_energy(self, audio_array: np.ndarray) -> float:
        if audio_array is None or len(audio_array) == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(audio_array.astype(np.float32)))))

    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        if embedding is None:
            return None

        normalized = np.asarray(embedding, dtype=np.float32).flatten()
        norm = float(np.linalg.norm(normalized))
        if norm == 0.0:
            return normalized
        return normalized / norm

    def build_registered_voice_profile(self, embeddings: List[np.ndarray]) -> Dict[str, Any]:
        normalized_embeddings = [
            self._normalize_embedding(embedding)
            for embedding in embeddings
            if embedding is not None and np.asarray(embedding).size > 0
        ]

        if not normalized_embeddings:
            return {
                "embeddings": [],
                "profile_embedding": None,
                "discarded_embeddings": 0,
            }

        if len(normalized_embeddings) == 1:
            return {
                "embeddings": normalized_embeddings,
                "profile_embedding": normalized_embeddings[0],
                "discarded_embeddings": 0,
            }

        centroid = self._normalize_embedding(np.mean(np.stack(normalized_embeddings), axis=0))
        centroid_similarities = np.array(
            [self.compare_embeddings(embedding, centroid) for embedding in normalized_embeddings],
            dtype=np.float32,
        )
        centroid_floor = max(
            self.registration_centroid_min_similarity,
            float(np.median(centroid_similarities) - 0.05),
        )
        filtered_embeddings = [
            embedding
            for embedding, similarity in zip(normalized_embeddings, centroid_similarities)
            if similarity >= centroid_floor
        ]

        minimum_keep = min(len(normalized_embeddings), 2)
        if len(filtered_embeddings) < minimum_keep:
            filtered_embeddings = normalized_embeddings

        if len(filtered_embeddings) >= 3:
            pairwise_scores = []
            for idx, embedding in enumerate(filtered_embeddings):
                neighbor_scores = [
                    self.compare_embeddings(embedding, other)
                    for other_idx, other in enumerate(filtered_embeddings)
                    if other_idx != idx
                ]
                pairwise_scores.append(float(np.mean(neighbor_scores)) if neighbor_scores else 0.0)

            pairwise_floor = max(
                self.registration_pairwise_min_similarity,
                float(np.median(pairwise_scores) - 0.05),
            )
            refined_embeddings = [
                embedding
                for embedding, score in zip(filtered_embeddings, pairwise_scores)
                if score >= pairwise_floor
            ]
            if len(refined_embeddings) >= 2:
                filtered_embeddings = refined_embeddings

        profile_embedding = self._normalize_embedding(np.mean(np.stack(filtered_embeddings), axis=0))
        discarded_embeddings = len(normalized_embeddings) - len(filtered_embeddings)
        print(
            f"[DIARIZATION] Registration profile built: kept {len(filtered_embeddings)}/{len(normalized_embeddings)} "
            f"embeddings, discarded {discarded_embeddings}"
        )

        return {
            "embeddings": filtered_embeddings,
            "profile_embedding": profile_embedding,
            "discarded_embeddings": discarded_embeddings,
        }

    def _is_registered_wearer_match(
        self,
        comparison_result: Dict[str, Any],
        recent_wearer: bool = False,
        full_context: bool = False,
    ) -> bool:
        avg_similarity = comparison_result.get("avg_similarity", 0.0)
        profile_similarity = comparison_result.get("profile_similarity", 0.0)
        median_similarity = comparison_result.get("median_similarity", 0.0)
        max_similarity = comparison_result.get("max_similarity", 0.0)
        similarities = np.array(comparison_result.get("all_similarities", []), dtype=float)
        strong_match_ratio = (
            float(np.mean(similarities >= self.wearer_multi_vote_sim_threshold))
            if similarities.size
            else 0.0
        )

        if full_context:
            avg_threshold = self.wearer_full_avg_threshold
            median_threshold = self.wearer_full_median_threshold
            median_max_threshold = self.wearer_full_median_max_threshold
            match_ratio_threshold = self.wearer_full_match_ratio_threshold
            match_ratio_max_threshold = self.wearer_full_match_ratio_max_threshold
            count_sim_threshold = self.wearer_full_count_sim_threshold
            count_ratio_threshold = self.wearer_full_count_ratio_threshold
        else:
            avg_threshold = self.wearer_chunk_avg_threshold
            median_threshold = self.wearer_chunk_median_threshold
            median_max_threshold = self.wearer_chunk_max_threshold
            match_ratio_threshold = self.wearer_chunk_match_ratio_threshold
            match_ratio_max_threshold = self.wearer_chunk_match_ratio_max_threshold
            count_sim_threshold = self.wearer_chunk_count_sim_threshold
            count_ratio_threshold = self.wearer_chunk_count_ratio_threshold

        count_ratio = (
            float(np.mean(similarities >= count_sim_threshold))
            if similarities.size
            else 0.0
        )

        if avg_similarity >= avg_threshold:
            return True

        if median_similarity >= median_threshold and max_similarity >= median_max_threshold:
            return True

        if strong_match_ratio >= match_ratio_threshold and max_similarity >= match_ratio_max_threshold:
            return True

        if count_ratio >= count_ratio_threshold:
            return True

        if recent_wearer and (
            avg_similarity >= avg_threshold - 0.02
            or profile_similarity >= avg_threshold - 0.02
        ) and max_similarity >= median_max_threshold - 0.02:
            return True

        return False

    def _get_tracked_speaker(self, speaker_tracking: Dict[str, Any], speaker_id: str) -> Optional[Dict[str, Any]]:
        if not speaker_tracking or not speaker_id:
            return None

        for tracked_speaker in speaker_tracking.get('speakers', []):
            if tracked_speaker.get('id') == speaker_id:
                return tracked_speaker
        return None

    def _is_recent_last_active_speaker(self, speaker_tracking: Dict[str, Any], speaker_id: str = None) -> bool:
        if not speaker_tracking:
            return False

        last_active_speaker_id = speaker_tracking.get('last_active_speaker_id')
        last_active_timestamp = speaker_tracking.get('last_active_timestamp')
        if not last_active_speaker_id or last_active_timestamp is None:
            return False

        if speaker_id and last_active_speaker_id != speaker_id:
            return False

        return (time.time() - last_active_timestamp) <= self.chunk_recent_speaker_window_sec

    def _extract_speech_metrics(
        self,
        whisper_segments: List[Any],
        chunk_duration: float,
    ) -> Dict[str, Any]:
        speech_segments: List[Tuple[float, float]] = []
        for segment in whisper_segments:
            start_time = max(0.0, float(getattr(segment, "start", 0.0) or 0.0))
            end_time = min(chunk_duration, float(getattr(segment, "end", start_time) or start_time))
            if end_time <= start_time:
                continue
            speech_segments.append((start_time, end_time))

        speech_duration = float(sum(end - start for start, end in speech_segments))
        longest_speech = float(
            max((end - start) for start, end in speech_segments)
        ) if speech_segments else 0.0
        speech_fraction = (speech_duration / chunk_duration) if chunk_duration > 0 else 0.0

        return {
            'speech_segments': speech_segments,
            'speech_duration': speech_duration,
            'speech_fraction': float(speech_fraction),
            'longest_speech': longest_speech,
        }

    def _build_embedding_audio_from_speech_segments(
        self,
        audio_array: np.ndarray,
        sample_rate: int,
        speech_segments: List[Tuple[float, float]],
    ) -> Optional[np.ndarray]:
        if audio_array is None or len(audio_array) == 0 or not speech_segments:
            return None

        padded_windows: List[np.ndarray] = []
        padding_samples = int(sample_rate * 0.12)

        for start_time, end_time in speech_segments:
            start_sample = max(0, int(start_time * sample_rate) - padding_samples)
            end_sample = min(len(audio_array), int(end_time * sample_rate) + padding_samples)
            if end_sample <= start_sample:
                continue

            window = audio_array[start_sample:end_sample].copy()
            if len(window) < int(sample_rate * 0.12):
                continue
            padded_windows.append(window)

        if not padded_windows:
            return None

        max_samples = int(sample_rate * 2.4)
        combined_windows: List[np.ndarray] = []
        combined_samples = 0
        for window in sorted(padded_windows, key=len, reverse=True):
            remaining = max_samples - combined_samples
            if remaining <= 0:
                break
            if len(window) <= remaining:
                combined_windows.append(window)
                combined_samples += len(window)
            else:
                combined_windows.append(window[:remaining])
                combined_samples += remaining

        if combined_windows:
            combined_audio = np.concatenate(combined_windows)
            if len(combined_audio) >= int(self.chunk_embedding_min_speech_duration * sample_rate):
                return combined_audio

        longest_window = max(padded_windows, key=len)
        min_samples = int(self.chunk_embedding_min_speech_duration * sample_rate)
        if len(longest_window) >= min_samples:
            return longest_window

        concatenated = np.concatenate(padded_windows)
        if len(concatenated) >= int(sample_rate * 0.20):
            fallback_max_samples = int(sample_rate * 1.8)
            return concatenated[:fallback_max_samples]

        return None

    def _should_skip_chunk_transcription(
        self,
        text: str,
        original_text: str,
        duration: float,
        rms_energy: float,
        avg_logprob: float = None,
        no_speech_prob: float = None,
        speech_duration: float = None,
        speech_fraction: float = None,
        longest_speech: float = None,
    ) -> bool:
        normalized_text = "".join((text or "").split())
        normalized_original = "".join((original_text or "").split())

        if rms_energy < self.chunk_min_rms_energy:
            print(f"[DIARIZATION] Skipping weak chunk (RMS too low: {rms_energy:.6f})")
            return True

        if avg_logprob is not None and avg_logprob < self.chunk_min_avg_logprob:
            print(f"[DIARIZATION] Skipping low-confidence chunk (avg_logprob={avg_logprob:.3f})")
            return True

        if no_speech_prob is not None and no_speech_prob > self.chunk_max_no_speech_prob:
            has_meaningful_text = len(normalized_original) >= 2
            has_strong_speech_span = (
                (speech_duration is not None and speech_duration >= 0.55)
                or (speech_fraction is not None and speech_fraction >= 0.20)
                or (longest_speech is not None and longest_speech >= 0.35)
            )
            confident_text = avg_logprob is not None and avg_logprob >= -0.85

            if not (has_meaningful_text and (has_strong_speech_span or confident_text)):
                print(f"[DIARIZATION] Skipping likely non-speech chunk (no_speech_prob={no_speech_prob:.3f})")
                return True

            print(
                f"[DIARIZATION] Keeping chunk despite high no_speech_prob={no_speech_prob:.3f} "
                f"(text/speech evidence looks valid)"
            )

        if speech_duration is not None and speech_duration < self.chunk_min_speech_duration:
            print(f"[DIARIZATION] Skipping sparse chunk (speech duration too short: {speech_duration:.3f}s)")
            return True

        if speech_fraction is not None and speech_fraction < self.chunk_min_speech_fraction:
            print(f"[DIARIZATION] Skipping sparse chunk (speech fraction too low: {speech_fraction:.3f})")
            return True

        if (
            duration <= 3.5
            and speech_duration is not None
            and speech_duration < 0.75
            and len(normalized_original) >= 12
        ):
            print(
                f"[DIARIZATION] Skipping dense text for sparse speech chunk "
                f"(speech={speech_duration:.3f}s, text_len={len(normalized_original)})"
            )
            return True

        if (
            duration <= 3.5
            and longest_speech is not None
            and longest_speech < 0.25
            and len(normalized_text) >= 4
        ):
            print(
                f"[DIARIZATION] Skipping bursty chunk transcription "
                f"(longest speech span={longest_speech:.3f}s)"
            )
            return True

        if duration <= 3.5 and len(normalized_text) <= 1:
            print(f"[DIARIZATION] Skipping extremely short chunk transcription: '{text}'")
            return True

        if normalized_text and len(set(normalized_text)) == 1 and len(normalized_text) >= 3:
            print(f"[DIARIZATION] Skipping repetitive chunk transcription: '{text}'")
            return True

        suspicious_patterns = [
            r"^[例這该則若於為與及之乎者也]{4,}$",
        ]
        if duration <= 3.5 and rms_energy < 0.006:
            for pattern in suspicious_patterns:
                if re.fullmatch(pattern, normalized_text) or re.fullmatch(pattern, normalized_original):
                    print(f"[DIARIZATION] Skipping suspicious chunk transcription: '{text}'")
                    return True

        return False

    def _update_tracked_speaker_embedding(self, tracked_speaker: Dict[str, Any], chunk_embedding: np.ndarray, alpha: float):
        old_emb = tracked_speaker['embedding']
        tracked_speaker['embedding'] = self._normalize_embedding(
            (1 - alpha) * old_emb + alpha * chunk_embedding
        )

    def _set_last_active_speaker(self, speaker_tracking: Dict[str, Any], speaker_id: str):
        if not speaker_tracking or not speaker_id:
            return
        speaker_tracking['last_active_speaker_id'] = speaker_id
        speaker_tracking['last_active_timestamp'] = time.time()

    def _match_or_create_tracked_speaker(
        self,
        speaker_tracking: Dict[str, Any],
        chunk_embedding: np.ndarray,
        speech_duration: float = None,
        speech_fraction: float = None,
    ) -> Dict[str, Any]:
        if not speaker_tracking or 'speakers' not in speaker_tracking:
            return {
                'speaker_id': 'SPEAKER_01',
                'is_wearer': False,
                'similarity': 0.0,
            }

        best_match_similarity = 0.0
        matched_speaker = None
        last_active_speaker_id = speaker_tracking.get('last_active_speaker_id')
        last_active_speaker = self._get_tracked_speaker(speaker_tracking, last_active_speaker_id)
        last_active_similarity = 0.0
        weak_switch_evidence = (
            (speech_duration is not None and speech_duration < self.chunk_min_speech_duration_for_switch)
            or (speech_fraction is not None and speech_fraction < self.chunk_min_speech_fraction_for_switch)
        )

        for known_speaker in speaker_tracking['speakers']:
            if known_speaker.get('is_wearer', False):
                continue

            known_emb = known_speaker.get('embedding')
            if known_emb is None:
                continue

            sim = self.compare_embeddings(chunk_embedding, known_emb)
            if sim > best_match_similarity:
                best_match_similarity = sim
                matched_speaker = known_speaker

            if known_speaker.get('id') == last_active_speaker_id:
                last_active_similarity = sim

        if (
            matched_speaker
            and last_active_speaker is not None
            and matched_speaker.get('id') != last_active_speaker_id
            and self._is_recent_last_active_speaker(speaker_tracking)
            and last_active_similarity >= self.chunk_hold_speaker_threshold
            and (
                weak_switch_evidence
                or (best_match_similarity - last_active_similarity) <= self.chunk_speaker_switch_margin
            )
        ):
            self._update_tracked_speaker_embedding(last_active_speaker, chunk_embedding, 0.05)
            speaker_tracking.pop('pending_new_speaker', None)
            self._set_last_active_speaker(speaker_tracking, last_active_speaker_id)
            print(
                f"[DIARIZATION] Keeping recent speaker {last_active_speaker_id} "
                f"(candidate={matched_speaker['id']}, best={best_match_similarity:.4f}, last={last_active_similarity:.4f})"
            )
            return {
                'speaker_id': last_active_speaker_id,
                'is_wearer': last_active_speaker.get('is_wearer', False),
                'similarity': last_active_similarity,
            }

        if matched_speaker and best_match_similarity >= self.chunk_speaker_match_threshold:
            self._update_tracked_speaker_embedding(
                matched_speaker, chunk_embedding, self.speaker_embedding_update_alpha
            )
            speaker_tracking.pop('pending_new_speaker', None)
            self._set_last_active_speaker(speaker_tracking, matched_speaker['id'])
            print(f"[DIARIZATION] Chunk matched to {matched_speaker['id']} - Similarity: {best_match_similarity:.4f}")
            return {
                'speaker_id': matched_speaker['id'],
                'is_wearer': matched_speaker.get('is_wearer', False),
                'similarity': best_match_similarity,
            }

        if matched_speaker and best_match_similarity >= self.chunk_speaker_soft_match_threshold:
            self._update_tracked_speaker_embedding(matched_speaker, chunk_embedding, 0.10)
            speaker_tracking.pop('pending_new_speaker', None)
            self._set_last_active_speaker(speaker_tracking, matched_speaker['id'])
            print(f"[DIARIZATION] Merged chunk with {matched_speaker['id']} (similarity: {best_match_similarity:.4f})")
            return {
                'speaker_id': matched_speaker['id'],
                'is_wearer': matched_speaker.get('is_wearer', False),
                'similarity': best_match_similarity,
            }

        non_wearer_speakers = [s for s in speaker_tracking['speakers'] if not s.get('is_wearer', False)]
        if not non_wearer_speakers:
            speaker_id = f"SPEAKER_{speaker_tracking.get('next_id', 1):02d}"
            speaker_tracking['speakers'].append({
                'id': speaker_id,
                'embedding': chunk_embedding,
                'is_wearer': False,
            })
            speaker_tracking['next_id'] = speaker_tracking.get('next_id', 1) + 1
            speaker_tracking.pop('pending_new_speaker', None)
            self._set_last_active_speaker(speaker_tracking, speaker_id)
            print(f"[DIARIZATION] Initial tracked speaker created: {speaker_id}")
            return {
                'speaker_id': speaker_id,
                'is_wearer': False,
                'similarity': best_match_similarity,
            }

        pending_new_speaker = speaker_tracking.get('pending_new_speaker')

        if pending_new_speaker is not None:
            pending_similarity = self.compare_embeddings(chunk_embedding, pending_new_speaker['embedding'])
            if pending_similarity >= self.chunk_pending_speaker_similarity:
                pending_new_speaker['count'] += 1
                pending_new_speaker['embedding'] = (
                    0.5 * pending_new_speaker['embedding'] + 0.5 * chunk_embedding
                )
                pending_new_speaker['best_similarity'] = max(
                    pending_new_speaker.get('best_similarity', 0.0),
                    best_match_similarity,
                )
            else:
                pending_new_speaker = None

        if pending_new_speaker is None:
            pending_new_speaker = {
                'embedding': chunk_embedding,
                'count': 1,
                'best_similarity': best_match_similarity,
            }
            speaker_tracking['pending_new_speaker'] = pending_new_speaker

        if last_active_speaker_id and (
            weak_switch_evidence
            or
            best_match_similarity >= self.chunk_hold_speaker_threshold
            or pending_new_speaker['count'] < self.chunk_new_speaker_confirmations
        ):
            print(
                f"[DIARIZATION] Holding previous speaker {last_active_speaker_id} "
                f"(best match {best_match_similarity:.4f}, pending confirmations {pending_new_speaker['count']}/{self.chunk_new_speaker_confirmations})"
            )
            self._set_last_active_speaker(speaker_tracking, last_active_speaker_id)
            return {
                'speaker_id': last_active_speaker_id,
                'is_wearer': False,
                'similarity': best_match_similarity,
            }

        next_id = speaker_tracking.get('next_id', 1)
        speaker_id = f'SPEAKER_{next_id:02d}'
        speaker_tracking['speakers'].append({
            'id': speaker_id,
            'embedding': chunk_embedding,
            'is_wearer': False,
        })
        speaker_tracking['next_id'] = next_id + 1
        speaker_tracking.pop('pending_new_speaker', None)
        self._set_last_active_speaker(speaker_tracking, speaker_id)
        print(f"[DIARIZATION] New speaker detected: {speaker_id} (best match was {best_match_similarity:.4f})")
        return {
            'speaker_id': speaker_id,
            'is_wearer': False,
            'similarity': best_match_similarity,
        }
        
    def _cleanup_memory(self):
        """Clean up memory to prevent OOM."""
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def extract_multiple_embeddings(self, audio_array: np.ndarray, sample_rate: int = 16000, 
                                   segment_duration: float = None) -> List[np.ndarray]:
        """
        Extract multiple 192-dim embeddings from audio by segmenting it.
        This creates a more robust voice profile by capturing variations.
        
        Args:
            audio_array: Audio as numpy array (float32, mono, normalized to [-1, 1])
            sample_rate: Sample rate in Hz (default: 16000)
            segment_duration: Duration of each segment in seconds (default: 1.5s)
            
        Returns:
            List of embeddings, each with shape (192,)
            
        Example:
            embeddings = pipeline.extract_multiple_embeddings(audio_array, 16000)
            # Returns: [emb1, emb2, emb3, ...] - multiple 192-dim embeddings
        """
        try:
            duration_sec = len(audio_array) / sample_rate
            segment_duration = segment_duration or self.registration_segment_duration
            print(f"[DIARIZATION] ========== MULTI-SAMPLE REGISTRATION ==========")
            print(f"[DIARIZATION] Total audio: {duration_sec:.2f}s ({len(audio_array)} samples)")
            print(f"[DIARIZATION] Segment duration: {segment_duration}s")
            
            # Ensure audio is mono
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)
            audio_array = audio_array.astype(np.float32)
            
            # DO NOT normalize whole audio here - normalize each segment individually
            # to match how conversation segments are processed!
            
            embeddings = []
            segment_samples = int(segment_duration * sample_rate)
            overlap_samples = int(self.registration_overlap_duration * sample_rate)
            
            start_idx = 0
            segment_num = 0
            
            while start_idx < len(audio_array):
                end_idx = min(start_idx + segment_samples, len(audio_array))
                segment_audio = audio_array[start_idx:end_idx].copy()  # Copy to avoid modifying original
                
                # Skip if segment too short
                if len(segment_audio) < sample_rate * self.registration_min_segment_duration:
                    print(f"[DIARIZATION] Segment {segment_num+1}: Too short, skipping")
                    break
                
                # Check segment energy BEFORE normalization
                rms_energy = np.sqrt(np.mean(segment_audio**2))
                if rms_energy < self.registration_min_rms_energy:
                    print(f"[DIARIZATION] Segment {segment_num+1}: Too quiet (RMS: {rms_energy:.6f}), skipping")
                    start_idx += segment_samples - overlap_samples
                    segment_num += 1
                    continue
                
                # IMPORTANT: Normalize EACH segment individually (same as conversation processing!)
                if np.max(np.abs(segment_audio)) > 0:
                    segment_audio = segment_audio / np.max(np.abs(segment_audio))
                
                # Extract embedding from this segment
                segment_duration_actual = len(segment_audio) / sample_rate
                print(f"[DIARIZATION] Segment {segment_num+1}: {segment_duration_actual:.2f}s, RMS: {rms_energy:.6f}")
                
                audio_tensor = torch.from_numpy(segment_audio).unsqueeze(0)
                
                with torch.no_grad():
                    embedding = self.speaker_model.encode_batch(audio_tensor)
                    embedding = self._normalize_embedding(embedding.squeeze().cpu().numpy())
                
                # Validate embedding
                if embedding.shape[0] == 192:
                    embeddings.append(embedding)
                    emb_stats = f"mean={np.mean(embedding):.4f}, std={np.std(embedding):.4f}"
                    print(f"[DIARIZATION] ✓ Segment {segment_num+1} embedding extracted ({emb_stats})")
                else:
                    print(f"[DIARIZATION] ✗ Segment {segment_num+1} invalid shape: {embedding.shape}")
                
                # Move to next segment with overlap
                start_idx += segment_samples - overlap_samples
                segment_num += 1
            
            if len(embeddings) == 0:
                print(f"[DIARIZATION] ERROR: No valid embeddings extracted!")
                return None
            
            print(f"[DIARIZATION] ========================================")
            print(f"[DIARIZATION] ✓ Extracted {len(embeddings)} embeddings total")
            print(f"[DIARIZATION] Each embedding shape: (192,)")
            
            # Calculate stats across embeddings
            embeddings_array = np.array(embeddings)
            print(f"[DIARIZATION] Embeddings array shape: {embeddings_array.shape}")
            print(f"[DIARIZATION] Overall mean: {np.mean(embeddings_array):.4f}")
            print(f"[DIARIZATION] Overall std: {np.std(embeddings_array):.4f}")
            
            # Show individual embedding stats for debugging
            for i, emb in enumerate(embeddings):
                print(f"[DIARIZATION]   Sample {i+1}: mean={np.mean(emb):.4f}, std={np.std(emb):.4f}, norm={np.linalg.norm(emb):.4f}")
            
            print(f"[DIARIZATION] ========================================")
            
            return embeddings
            
        except Exception as e:
            print(f"[DIARIZATION] ERROR extracting multiple embeddings: {e}")
            import traceback
            print(f"[DIARIZATION] Traceback: {traceback.format_exc()}")
            return None
    
    def extract_speaker_embedding(self, audio_array: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """
        Extract 192-dimensional speaker embedding from audio.
        This function can be called from ar_glasses_server to get embeddings for voice registration.
        
        Args:
            audio_array: Audio as numpy array (float32, mono, normalized to [-1, 1])
            sample_rate: Sample rate in Hz (default: 16000)
            
        Returns:
            np.ndarray: 192-dimensional embedding, or None if failed
            
        Example:
            embedding = diarization_pipeline.extract_speaker_embedding(audio_array, 16000)
            # Returns: np.array with shape (192,)
        """
        try:
            print(f"[DIARIZATION] Extracting 192-dim speaker embedding...")
            print(f"[DIARIZATION] Audio: {len(audio_array)} samples at {sample_rate}Hz")
            
            # Ensure audio is mono
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)
            
            # Convert to float32
            audio_array = audio_array.astype(np.float32)
            
            # Check audio energy BEFORE normalization
            rms_energy = np.sqrt(np.mean(audio_array**2))
            if len(audio_array) / sample_rate > 3.0:  # Only log for longer audio
                print(f"[DIARIZATION] Audio RMS energy (before norm): {rms_energy:.6f}")
            
            # Normalize to [-1, 1] range
            if np.max(np.abs(audio_array)) > 0:
                audio_array = audio_array / np.max(np.abs(audio_array))
                if len(audio_array) / sample_rate > 3.0:  # Only log for longer audio
                    print(f"[DIARIZATION] Audio normalized to range: [{np.min(audio_array):.4f}, {np.max(audio_array):.4f}]")
            
            # Convert to torch tensor [1, samples]
            audio_tensor = torch.from_numpy(audio_array).unsqueeze(0)
            
            # Extract embedding using SpeechBrain ECAPA-TDNN
            with torch.no_grad():
                embedding = self.speaker_model.encode_batch(audio_tensor)
                embedding = self._normalize_embedding(embedding.squeeze().cpu().numpy())  # Shape: (192,)
            
            if len(audio_array) / sample_rate > 3.0:  # Only log for longer audio
                print(f"[DIARIZATION] ✓ Embedding extracted successfully!")
                print(f"[DIARIZATION] Embedding shape: {embedding.shape}")
                print(f"[DIARIZATION] Embedding stats: mean={np.mean(embedding):.4f}, std={np.std(embedding):.4f}, norm={np.linalg.norm(embedding):.4f}")
            
            return embedding
            
        except Exception as e:
            print(f"[DIARIZATION] ERROR extracting embedding: {e}")
            import traceback
            print(f"[DIARIZATION] Traceback: {traceback.format_exc()}")
            return None
        
    def _manage_cache(self):
        """Manage cache size to prevent memory overflow."""
        with self.cache_lock:
            if len(self.audio_cache) > self.max_cache_size:
                # Remove oldest entries
                keys_to_remove = list(self.audio_cache.keys())[:-self.max_cache_size]
                for key in keys_to_remove:
                    del self.audio_cache[key]
                self._cleanup_memory()

    def preprocess_audio(self, audio_segment: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Preprocess audio for maximum transcription accuracy."""
        # Ensure audio is float32 (required by Whisper)
        if audio_segment.dtype != np.float32:
            audio_segment = audio_segment.astype(np.float32)
        
        # Normalize audio (simple normalization - original approach)
        if np.max(np.abs(audio_segment)) > 0:
            audio_segment = audio_segment / np.max(np.abs(audio_segment))
        
        return audio_segment

    def transcribe_segment(self, audio_segment: np.ndarray, sample_rate: int = 16000, 
                          transcription_lang: str = None, translation_lang: str = None,
                          is_chunk: bool = False) -> Dict[str, Any]:
        """Transcribe a single audio segment using optimized Whisper settings.
        
        Args:
            audio_segment: Audio data to transcribe
            sample_rate: Sample rate of audio
            transcription_lang: Language for transcription (uses default if None)
            translation_lang: Target language for translation (no translation if None)
            is_chunk: If True, use faster settings for real-time processing
        
        Returns:
            Dictionary containing:
                - 'text': Final text (translated if needed)
                - 'original_text': Original transcription
                - 'translated': Boolean indicating if translation was performed
                - 'source_lang': Source language
                - 'target_lang': Target language (if translated)
        """
        try:
            # Use provided language or default
            source_lang = transcription_lang if transcription_lang else self.transcription_lang
            whisper_language = self._normalize_whisper_language(source_lang)
            initial_prompt = None
            
            # Preprocess audio for maximum accuracy
            audio_segment = self.preprocess_audio(audio_segment, sample_rate)
            
            # Use faster settings for chunks (real-time processing)
            transcribe_options = dict(
                self.chunk_transcribe_options if is_chunk else self.full_transcribe_options
            )
            transcribe_options["task"] = "transcribe"
            if whisper_language is not None:
                transcribe_options["language"] = whisper_language
            if initial_prompt is not None:
                transcribe_options["initial_prompt"] = initial_prompt

            segments, info = self.whisper_model.transcribe(
                audio_segment,
                **transcribe_options,
            )
            
            segments = list(segments)
            speech_metrics = self._extract_speech_metrics(
                segments,
                len(audio_segment) / sample_rate if sample_rate > 0 else 0.0,
            )

            # Combine all segments into one text
            original_text = " ".join([segment.text for segment in segments]).strip()
            avg_logprob = (
                float(np.mean([getattr(segment, "avg_logprob", -0.5) for segment in segments]))
                if segments
                else None
            )
            no_speech_prob = (
                float(max([getattr(segment, "no_speech_prob", 0.0) for segment in segments]))
                if segments
                else None
            )
            
            resolved_source_lang = self._resolve_source_language(source_lang, info)

            original_text = self._strip_prompt_echo(original_text)

            # Post-process for language output
            original_text = self.ensure_cantonese_output(original_text, info)
            
            # Prepare result
            result = {
                'text': original_text,
                'original_text': original_text,
                'translated': False,
                'source_lang': resolved_source_lang,
                'target_lang': None,
                'avg_logprob': avg_logprob,
                'no_speech_prob': no_speech_prob,
                'speech_duration': speech_metrics['speech_duration'],
                'speech_fraction': speech_metrics['speech_fraction'],
                'longest_speech': speech_metrics['longest_speech'],
                'speech_segments': speech_metrics['speech_segments'],
            }
            
            # Apply translation if needed (and target language is provided)
            if self.translator and translation_lang:
                translation_result = self.translator.translate_text(
                    original_text,
                    resolved_source_lang,
                    translation_lang
                )
                
                if translation_result.get('translated'):
                    result['text'] = translation_result['text']
                    result['translated'] = True
                    result['target_lang'] = translation_lang
                    print(f"[DIARIZATION] Translated: '{original_text[:30]}...' -> '{result['text'][:30]}...'")
                elif translation_result.get('error'):
                    print(f"[DIARIZATION] Translation error: {translation_result['error']}")
                    # Keep original text if translation fails
            
            return result
            
        except Exception as e:
            print(f"[DIARIZATION] Error transcribing segment: {e}")
            return {
                'text': '',
                'original_text': '',
                'translated': False,
                'source_lang': source_lang if 'source_lang' in locals() else self.transcription_lang,
                'target_lang': None,
                'avg_logprob': None,
                'no_speech_prob': None,
            }

    def compare_with_multiple_embeddings(self, segment_embedding: np.ndarray, 
                                         registered_embeddings: List[np.ndarray],
                                         profile_embedding: np.ndarray = None) -> Dict[str, Any]:
        """
        Compare a segment embedding with multiple registered embeddings.
        Uses voting and averaging for robust detection.
        
        Args:
            segment_embedding: Embedding from current segment (192,)
            registered_embeddings: List of registered embeddings [(192,), (192,), ...]
            
        Returns:
            Dictionary with similarity metrics:
                - 'max_similarity': Best match score
                - 'avg_similarity': Average across all samples
                - 'median_similarity': Median score
                - 'match_count': Number of samples above threshold
        """
        similarities = []
        normalized_segment_embedding = self._normalize_embedding(segment_embedding)
        
        for i, reg_emb in enumerate(registered_embeddings):
            sim = self.compare_embeddings(normalized_segment_embedding, reg_emb)
            similarities.append(sim)
        
        similarities_array = np.array(similarities)
        
        # Calculate different metrics
        max_sim = np.max(similarities_array)
        avg_sim = np.mean(similarities_array)
        median_sim = np.median(similarities_array)
        profile_similarity = (
            self.compare_embeddings(normalized_segment_embedding, profile_embedding)
            if profile_embedding is not None
            else median_sim
        )
        
        # Count how many samples are above threshold (0.62 - lowered for more leniency)
        match_count = np.sum(similarities_array >= self.wearer_multi_vote_sim_threshold)
        moderate_match_count = np.sum(
            similarities_array >= min(self.wearer_chunk_count_sim_threshold, self.wearer_full_count_sim_threshold)
        )
        
        return {
            'max_similarity': float(max_sim),
            'avg_similarity': float(avg_sim),
            'median_similarity': float(median_sim),
            'match_count': int(match_count),
            'strong_match_ratio': float(match_count / len(similarities)) if similarities else 0.0,
            'moderate_match_count': int(moderate_match_count),
            'moderate_match_ratio': float(moderate_match_count / len(similarities)) if similarities else 0.0,
            'total_samples': len(similarities),
            'all_similarities': similarities,
            'profile_similarity': float(profile_similarity),
        }
    
    def compare_embeddings(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compare two speaker embeddings using cosine similarity.
        
        Args:
            embedding1: First embedding (192-dim)
            embedding2: Second embedding (192-dim)
            
        Returns:
            float: Cosine similarity score (0-1), where higher means more similar
        """
        try:
            # Ensure embeddings are 1D
            emb1 = embedding1.flatten()
            emb2 = embedding2.flatten()
            
            # Compute cosine similarity
            dot_product = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            
            # Ensure similarity is in [0, 1] range
            similarity = np.clip(similarity, 0.0, 1.0)
            
            return float(similarity)
            
        except Exception as e:
            print(f"[DIARIZATION] Error comparing embeddings: {e}")
            return 0.0
    
    def ensure_cantonese_output(self, text: str, info) -> str:
        """Ensure the output is in Cantonese characters, not English."""
        if not text:
            return text
        
        # Check if the detected language is actually Cantonese
        detected_lang = getattr(info, 'language', 'unknown')
        lang_confidence = getattr(info, 'language_probability', 0.0)
        
        print(f"[DIARIZATION] Detected language: {detected_lang} (confidence: {lang_confidence:.3f})")
        print(f"[DIARIZATION] Raw transcription: '{text}'")
        
        # Check if text contains Cantonese characters (CJK Unified Ideographs)
        has_cantonese = any('\u4e00' <= char <= '\u9fff' for char in text)
        
        if has_cantonese:
            print(f"[DIARIZATION] Text contains Cantonese characters - keeping as-is")
            return text
        
        has_latin = any(('a' <= char.lower() <= 'z') for char in text)
        if has_latin and self.allow_english:
            print(f"[DIARIZATION] English output detected and allowed - keeping as-is")
            return text

        # If we get English output when expecting Cantonese, try to force Cantonese
        print(f"[DIARIZATION] Text appears to be English, but we expected Cantonese")
        print(f"[DIARIZATION] This might be due to audio quality or model limitations")
        
        # Return the text as-is for now, but log the issue
        return text

    def process_audio_array(self, audio_array: np.ndarray, sample_rate: int = 16000, 
                           registered_voices: Dict[str, Any] = None,
                           transcription_lang: str = None, 
                           translation_lang: str = None,
                           is_chunk: bool = False,
                           speaker_tracking: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process audio array for speaker diarization and transcription.
        
        Args:
            audio_array: Audio data as numpy array
            sample_rate: Sample rate in Hz
            registered_voices: Dictionary of registered voice embeddings {voice_id: {'embedding': np.array, ...}}
            transcription_lang: Language for transcription (uses default if None)
            translation_lang: Target language for translation (no translation if None)
            is_chunk: If True, this is a real-time chunk (optimize for speed while keeping diarization)
        """
        try:
            # Only log for full audio, not chunks (will log after if segments found)
            if not is_chunk:
                chunk_mode = "FULL"
                print(f"[DIARIZATION] Processing audio array ({chunk_mode}): {len(audio_array)} samples, {sample_rate} Hz")
                print(f"[DIARIZATION] Transcription language: {transcription_lang or self.transcription_lang}")
                print(f"[DIARIZATION] Translation language: {translation_lang or 'None (no translation)'}")
                
                # Debug: Check what we received
                print(f"[DIARIZATION] DEBUG: registered_voices parameter = {type(registered_voices)}")
                print(f"[DIARIZATION] DEBUG: registered_voices is None? {registered_voices is None}")
                if registered_voices:
                    print(f"[DIARIZATION] DEBUG: registered_voices keys = {list(registered_voices.keys())}")
                    print(f"[DIARIZATION] DEBUG: Number of voices = {len(registered_voices)}")
            
            # Check if we have a registered voice (wearer's voice)
            has_registered_voice = registered_voices and len(registered_voices) > 0
            if not is_chunk:
                if has_registered_voice:
                    print(f"[DIARIZATION] ✓ Registered voice detected - will identify wearer's segments")
                else:
                    print(f"[DIARIZATION] ✗ No registered voice - processing normally")
            
            # Ensure audio is mono and float32
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)
            audio_array = audio_array.astype(np.float32)
            original_audio_array = audio_array.copy()
            raw_rms_energy = self._compute_rms_energy(audio_array)
            raw_peak_amplitude = float(np.max(np.abs(audio_array))) if len(audio_array) > 0 else 0.0
            
            # Audio preprocessing for better diarization
            # Normalize audio to improve detection
            if np.max(np.abs(audio_array)) > 0:
                audio_array = audio_array / np.max(np.abs(audio_array))
            
            if not is_chunk:
                print(f"[DIARIZATION] Audio preprocessed: {len(audio_array)} samples, max={np.max(audio_array):.3f}")
            
            # FAST PATH: For real-time chunks, skip diarization and use direct transcription
            if is_chunk:
                if raw_rms_energy < self.chunk_min_rms_energy or raw_peak_amplitude < self.chunk_min_peak_amplitude:
                    print(
                        f"[DIARIZATION] Skipping weak chunk before transcription "
                        f"(RMS={raw_rms_energy:.6f}, peak={raw_peak_amplitude:.6f})"
                    )
                    return {
                        'segments': [],
                        'total_duration': len(audio_array) / sample_rate,
                        'speaker_count': 0,
                        'processing_method': 'fast_chunk_transcription',
                        'is_chunk': True
                    }

                # Skip diarization for chunks - just transcribe the whole chunk directly
                # This is MUCH faster for real-time processing
                transcription_result = self.transcribe_segment(
                    audio_array,
                    sample_rate,
                    transcription_lang=transcription_lang,
                    translation_lang=translation_lang,
                    is_chunk=True  # Use fast settings
                )
                
                text = transcription_result.get('text', '') if transcription_result else ''
                original_text = transcription_result.get('original_text', text) if transcription_result else text
                chunk_duration = len(audio_array) / sample_rate
                
                # Skip only completely empty transcriptions (allow single characters/words)
                if not text or len(text.strip()) < 1:
                    return {
                        'segments': [],
                        'total_duration': len(audio_array) / sample_rate,
                        'speaker_count': 0,
                        'processing_method': 'fast_chunk_transcription',
                        'is_chunk': True
                    }

                if self._should_skip_chunk_transcription(
                    text,
                    original_text,
                    chunk_duration,
                    raw_rms_energy,
                    transcription_result.get('avg_logprob'),
                    transcription_result.get('no_speech_prob'),
                    transcription_result.get('speech_duration'),
                    transcription_result.get('speech_fraction'),
                    transcription_result.get('longest_speech'),
                ):
                    return {
                        'segments': [],
                        'total_duration': chunk_duration,
                        'speaker_count': 0,
                        'processing_method': 'fast_chunk_transcription',
                        'is_chunk': True
                    }
                
                # Filter out hallucination segments - skip entire segment if detected
                # Check both text and original_text, and normalize whitespace for matching
                skip_segment_patterns = [
                    "请不吝点赞 订阅 转发 打赏支持明镜与点点栏目",
                    "请不吝点赞订阅转发打赏支持明镜与点点栏目",  # Without spaces
                    "明镜与点点栏目",  # Key unique phrase from the hallucination
                    "谢谢大家"
                ]
                
                # Normalize text for comparison (remove extra whitespace)
                text_normalized = " ".join(text.split())
                original_text_normalized = " ".join(original_text.split())
                
                for pattern in skip_segment_patterns:
                    # Check both normalized text fields
                    if pattern in text_normalized or pattern in original_text_normalized:
                        print(f"[DIARIZATION] Skipping chunk segment containing hallucination: '{pattern}'")
                        print(f"[DIARIZATION] Detected in text: '{text[:100]}'")
                        print(f"[DIARIZATION] Detected in original_text: '{original_text[:100]}'")
                        return {
                            'segments': [],
                            'total_duration': len(audio_array) / sample_rate,
                            'speaker_count': 0,
                            'processing_method': 'fast_chunk_transcription',
                            'is_chunk': True
                        }
                
                # Speaker identification: Track speakers across chunks for consistent IDs
                is_wearer = False
                voice_similarity = 0.0
                speaker_id = 'SPEAKER_01'
                speech_duration = transcription_result.get('speech_duration', 0.0)
                speech_fraction = transcription_result.get('speech_fraction', 0.0)
                speech_segments = transcription_result.get('speech_segments', [])

                chunk_embedding = None
                embedding_audio = self._build_embedding_audio_from_speech_segments(
                    original_audio_array,
                    sample_rate,
                    speech_segments,
                )
                if embedding_audio is not None:
                    chunk_embedding = self.extract_speaker_embedding(embedding_audio, sample_rate)
                elif self._is_recent_last_active_speaker(speaker_tracking):
                    last_speaker = self._get_tracked_speaker(
                        speaker_tracking,
                        speaker_tracking.get('last_active_speaker_id'),
                    )
                    if last_speaker is not None:
                        speaker_id = last_speaker.get('id', speaker_id)
                        is_wearer = last_speaker.get('is_wearer', False)
                        print(
                            f"[DIARIZATION] Reusing recent speaker {speaker_id} "
                            f"because chunk lacked enough voiced audio for embedding"
                        )

                if chunk_embedding is not None:
                    if has_registered_voice:
                        _, voice_data = next(iter(registered_voices.items()))
                        registered_embeddings = voice_data.get('embeddings')
                        profile_embedding = voice_data.get('profile_embedding')
                        registered_embedding_single = voice_data.get('embedding')
                        recent_wearer = self._is_recent_last_active_speaker(speaker_tracking, 'SPEAKER_00')

                        if registered_embeddings is not None and len(registered_embeddings) >= 1 and profile_embedding is not None:
                            result = self.compare_with_multiple_embeddings(
                                chunk_embedding, registered_embeddings, profile_embedding
                            )
                            profile_sim = result['profile_similarity']
                            median_sim = result['median_similarity']
                            max_sim = result['max_similarity']
                            voice_similarity = profile_sim

                            print(
                                f"[DIARIZATION] Chunk wearer comparison: "
                                f"profile={profile_sim:.4f}, median={median_sim:.4f}, max={max_sim:.4f}"
                            )

                            if self._is_registered_wearer_match(result, recent_wearer=recent_wearer):
                                is_wearer = True
                                speaker_id = 'SPEAKER_00'
                                if speaker_tracking and 'speakers' in speaker_tracking:
                                    wearer_exists = any(s.get('id') == 'SPEAKER_00' for s in speaker_tracking['speakers'])
                                    if not wearer_exists:
                                        speaker_tracking['speakers'].append({
                                            'id': 'SPEAKER_00',
                                            'embedding': chunk_embedding,
                                            'is_wearer': True
                                        })
                                    self._set_last_active_speaker(speaker_tracking, speaker_id)
                                print(f"[DIARIZATION] Chunk identified as WEARER (SPEAKER_00) - Profile similarity: {profile_sim:.4f}")
                            else:
                                tracked_result = self._match_or_create_tracked_speaker(
                                    speaker_tracking,
                                    chunk_embedding,
                                    speech_duration=speech_duration,
                                    speech_fraction=speech_fraction,
                                )
                                speaker_id = tracked_result['speaker_id']
                                voice_similarity = tracked_result['similarity']
                        elif registered_embedding_single is not None:
                            similarity = self.compare_embeddings(chunk_embedding, registered_embedding_single)
                            voice_similarity = similarity

                            if similarity >= self.wearer_single_threshold_chunk or (recent_wearer and similarity >= self.wearer_recent_single_match_threshold):
                                is_wearer = True
                                speaker_id = 'SPEAKER_00'
                                if speaker_tracking and 'speakers' in speaker_tracking:
                                    wearer_exists = any(s.get('id') == 'SPEAKER_00' for s in speaker_tracking['speakers'])
                                    if not wearer_exists:
                                        speaker_tracking['speakers'].append({
                                            'id': 'SPEAKER_00',
                                            'embedding': chunk_embedding,
                                            'is_wearer': True
                                        })
                                    self._set_last_active_speaker(speaker_tracking, speaker_id)
                                print(f"[DIARIZATION] Chunk identified as WEARER (SPEAKER_00) - Similarity: {similarity:.4f}")
                            else:
                                tracked_result = self._match_or_create_tracked_speaker(
                                    speaker_tracking,
                                    chunk_embedding,
                                    speech_duration=speech_duration,
                                    speech_fraction=speech_fraction,
                                )
                                speaker_id = tracked_result['speaker_id']
                                voice_similarity = tracked_result['similarity']
                        else:
                            tracked_result = self._match_or_create_tracked_speaker(
                                speaker_tracking,
                                chunk_embedding,
                                speech_duration=speech_duration,
                                speech_fraction=speech_fraction,
                            )
                            speaker_id = tracked_result['speaker_id']
                            voice_similarity = tracked_result['similarity']
                    else:
                        tracked_result = self._match_or_create_tracked_speaker(
                            speaker_tracking,
                            chunk_embedding,
                            speech_duration=speech_duration,
                            speech_fraction=speech_fraction,
                        )
                        speaker_id = tracked_result['speaker_id']
                        is_wearer = tracked_result['is_wearer']
                        voice_similarity = tracked_result['similarity']
                
                # Create single segment for the chunk
                segment_data = {
                    'speaker_id': speaker_id,
                    'transcription': text,
                    'start': 0.0,
                    'end': len(audio_array) / sample_rate,
                    'duration': len(audio_array) / sample_rate,
                    'confidence': 0.85,
                    'original_text': transcription_result.get('original_text', text),
                    'translated': transcription_result.get('translated', False),
                    'source_lang': transcription_result.get('source_lang', transcription_lang or self.transcription_lang),
                    'is_wearer': is_wearer,
                    'voice_similarity': voice_similarity
                }
                
                if transcription_result.get('target_lang'):
                    segment_data['target_lang'] = transcription_result.get('target_lang')
                
                return {
                    'segments': [segment_data],
                    'total_duration': len(audio_array) / sample_rate,
                    'speaker_count': 1,
                    'processing_method': 'fast_chunk_transcription',
                    'is_chunk': True
                }
            
            # FULL PATH: For full audio, use full diarization
            # Create temporary file for diarization (required by pyannote)
            temp_file = f"temp_audio_{int(time.time() * 1000)}.wav"
            sf.write(temp_file, audio_array, sample_rate)
            
            try:
                # Perform speaker diarization
                print("[DIARIZATION] Running speaker diarization...")
                diarization = self.diarization_pipeline(temp_file)
                
                # Debug: Print all detected speakers and their segments
                unique_speakers = set()
                all_segments = []
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    unique_speakers.add(speaker)
                    all_segments.append({
                        'speaker': speaker,
                        'start': turn.start,
                        'end': turn.end,
                        'duration': turn.end - turn.start
                    })
                
                # Only log diarization details for full audio or when segments found
                if not is_chunk or len(all_segments) > 0:
                    print(f"[DIARIZATION] Detected speakers: {sorted(unique_speakers)}")
                    print(f"[DIARIZATION] Total diarization segments: {len(all_segments)}")
                    if len(all_segments) > 0:
                        print(f"[DIARIZATION] All diarization segments:")
                        for i, seg in enumerate(all_segments):
                            print(f"  Segment {i+1}: {seg['speaker']} ({seg['start']:.2f}s - {seg['end']:.2f}s, {seg['duration']:.2f}s)")
                    
                    # Calculate total coverage
                    total_audio_duration = len(audio_array) / sample_rate
                    covered_duration = sum(seg['duration'] for seg in all_segments)
                    coverage_percent = (covered_duration / total_audio_duration) * 100
                    print(f"[DIARIZATION] Audio coverage: {covered_duration:.2f}s / {total_audio_duration:.2f}s ({coverage_percent:.1f}%)")
                
                # If no speakers detected, log warning only for full audio (not chunks)
                if len(all_segments) == 0:
                    if not is_chunk:
                        print("[DIARIZATION] WARNING: No speakers detected by diarization!")
                        print("[DIARIZATION] This could be due to:")
                        print("[DIARIZATION] - Audio too short or too quiet")
                        print("[DIARIZATION] - Diarization model issues")
                        print("[DIARIZATION] - Audio format problems")
                    # For chunks, completely silent - no logging at all
                
                # Process each speaker segment with optimizations
                segments = []
                segment_count = 0
                skipped_short = 0
                skipped_energy = 0
                skipped_empty = 0
                
                print(f"[DIARIZATION] Processing {len(all_segments)} diarization segments...")
                
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    start_time = turn.start
                    end_time = turn.end
                    duration = end_time - start_time
                    
                    # OPTIMIZATION 1: Skip very short segments (prevents noise/hallucination speakers)
                    if duration < 0.3:  # Increased from 0.05s to 0.3s - skip very short segments
                        print(f"[OPTIMIZED] Skipping very short segment: {duration:.2f}s (likely noise)")
                        skipped_short += 1
                        continue
                    
                    if duration > 15.0:  # Increased limit for longer conversations
                        print(f"[DIARIZATION] Skipping very long segment: {duration:.2f}s")
                        continue
                    
                    # Extract audio segment
                    start_sample = int(start_time * sample_rate)
                    end_sample = int(end_time * sample_rate)
                    segment_audio = audio_array[start_sample:end_sample]
                    
                    # OPTIMIZATION 3: Check if segment has enough energy
                    if len(segment_audio) == 0:
                        continue
                    
                    # Calculate RMS energy (skip quiet segments that are likely noise)
                    rms_energy = np.sqrt(np.mean(segment_audio**2))
                    if rms_energy < 0.002:  # Increased from 0.0005 to 0.002 - skip quiet segments
                        print(f"[OPTIMIZED] Skipping low energy segment: {rms_energy:.6f} (likely background noise)")
                        skipped_energy += 1
                        continue
                    
                    # Memory cleanup before processing each segment
                    self._manage_cache()
                    self._cleanup_memory()
                    
                    #  Use cached transcription if available
                    segment_key = f"{start_time:.2f}_{end_time:.2f}_{speaker}"
                    cached_text = None
                    
                    with self.cache_lock:
                        if segment_key in self.audio_cache:
                            cached_text = self.audio_cache[segment_key]
                    
                    transcription_result = None
                    if cached_text is not None:
                        # Cached text is just a string, convert to result dict
                        transcription_result = {
                            'text': cached_text,
                            'original_text': cached_text,
                            'translated': False,
                            'source_lang': self.transcription_lang,
                            'target_lang': None
                        }
                        print(f"[DIARIZATION] Using cached transcription for segment {segment_count + 1}")
                    else:
                        # Transcribe segment (returns dict with translation info)
                        print(f"[DIARIZATION] Transcribing segment: {start_time:.2f}s - {end_time:.2f}s")
                        transcription_result = self.transcribe_segment(
                            segment_audio, 
                            sample_rate,
                            transcription_lang=transcription_lang,
                            translation_lang=translation_lang,
                            is_chunk=False  # Full audio uses normal settings
                        )
                        
                        # Cache the text result
                        with self.cache_lock:
                            self.audio_cache[segment_key] = transcription_result.get('text', '')
                            # Limit cache size
                            if len(self.audio_cache) > 100:
                                # Remove oldest entries
                                oldest_key = next(iter(self.audio_cache))
                                del self.audio_cache[oldest_key]
                    
                    # Extract text from result
                    text = transcription_result.get('text', '') if transcription_result else ''
                    original_text = transcription_result.get('original_text', text) if transcription_result else text
                    
                    # Filter out empty or very short transcriptions
                    if not text or len(text.strip()) < 2:  # Increased from 1 to 2 - skip single character
                        print(f"[OPTIMIZED] Skipping empty/short text: '{text}'")
                        skipped_empty += 1
                        continue
                    
                    # Filter out hallucination segments - skip entire segment if detected
                    # Check both text and original_text, and normalize whitespace for matching
                    skip_segment_patterns = [
                        "请不吝点赞 订阅 转发 打赏支持明镜与点点栏目",
                        "请不吝点赞订阅转发打赏支持明镜与点点栏目",  # Without spaces
                        "明镜与点点栏目",  # Key unique phrase from the hallucination
                        "谢谢大家"
                    ]
                    
                    # Normalize text for comparison (remove extra whitespace)
                    text_normalized = " ".join(text.split())
                    original_text_normalized = " ".join(original_text.split())
                    
                    should_skip = False
                    for pattern in skip_segment_patterns:
                        # Check both normalized text fields
                        if pattern in text_normalized or pattern in original_text_normalized:
                            print(f"[DIARIZATION] Skipping segment containing hallucination: '{pattern}'")
                            print(f"[DIARIZATION] Detected in text: '{text[:100]}'")
                            print(f"[DIARIZATION] Detected in original_text: '{original_text[:100]}'")
                            skipped_empty += 1
                            should_skip = True
                            break
                    
                    if should_skip:
                        continue
                    
                    # Filter out hallucination text from the segment (not the whole segment)
                    hallucination_patterns = [
                        "字幕由 Amara.org 社群提供",
                        "字幕由Amara.org社群提供",
                        "Amara.org",
                        "字幕由",
                        "社群提供",
                        "中文字幕",
                        "李宗盛",
                        "嗯",  # Um/uh sounds
                        "啊",  # Ah sounds
                        "呃",  # Uh sounds
                        "哦",  # Oh sounds
                        "喔",  # Oh sounds (variant)
                    ]
                    
                    original_text = text
                    for pattern in hallucination_patterns:
                        if pattern in text:
                            text = text.replace(pattern, "").strip()
                            print(f"[DIARIZATION] Removed hallucination '{pattern}' from segment")
                    
                    # Clean up spacing
                    text = " ".join(text.split())
                    
                    # If after removing hallucinations there's nothing left, skip the segment
                    if not text or len(text.strip()) < 2:
                        print(f"[DIARIZATION] Segment was entirely hallucination/noise: '{original_text}'")
                        skipped_empty += 1
                        continue
                    
                    # Additional filter: Skip if text is only punctuation or symbols
                    if text.strip() and all(not c.isalnum() for c in text.strip()):
                        print(f"[DIARIZATION] Skipping non-alphanumeric text: '{text}'")
                        skipped_empty += 1
                        continue
                    
                    # Compare with registered voice if available
                    is_wearer = False
                    voice_similarity = 0.0
                    
                    if has_registered_voice:
                        # Extract embedding from this segment
                        segment_embedding = self.extract_speaker_embedding(segment_audio, sample_rate)
                        
                        if segment_embedding is not None:
                            # Get the wearer's registered voice (only one voice)
                            voice_id, voice_data = next(iter(registered_voices.items()))
                            
                            # Check if using multi-sample registration
                            registered_embeddings = voice_data.get('embeddings')  # List of embeddings
                            registered_embedding_single = voice_data.get('embedding')  # Single embedding (legacy)
                            
                            if registered_embeddings is not None and len(registered_embeddings) > 1:
                                # Multi-sample comparison
                                print(f"[DIARIZATION] Comparing with {len(registered_embeddings)} registered samples")
                                
                                result = self.compare_with_multiple_embeddings(segment_embedding, registered_embeddings)
                                
                                max_sim = result['max_similarity']
                                avg_sim = result['avg_similarity']
                                median_sim = result['median_similarity']
                                match_count = result['match_count']
                                total_samples = result['total_samples']
                                
                                # Use median similarity for decision (more robust than average)
                                similarity = median_sim
                                voice_similarity = similarity
                                
                                # Get current segment embedding stats for comparison
                                seg_emb_mean = np.mean(segment_embedding)
                                seg_emb_std = np.std(segment_embedding)
                                seg_emb_norm = np.linalg.norm(segment_embedding)
                                
                                print(f"[DIARIZATION] ========== VOICE COMPARISON DEBUG ==========")
                                print(f"[DIARIZATION] Current segment embedding: mean={seg_emb_mean:.4f}, std={seg_emb_std:.4f}, norm={seg_emb_norm:.4f}")
                                print(f"[DIARIZATION] Comparing with {total_samples} registered embeddings:")
                                
                                # Show first registered embedding stats for comparison
                                reg_emb_0 = registered_embeddings[0]
                                reg_mean = np.mean(reg_emb_0)
                                reg_std = np.std(reg_emb_0)
                                reg_norm = np.linalg.norm(reg_emb_0)
                                print(f"[DIARIZATION] Registered sample 1: mean={reg_mean:.4f}, std={reg_std:.4f}, norm={reg_norm:.4f}")
                                
                                print(f"[DIARIZATION] Multi-sample results:")
                                print(f"[DIARIZATION]   Max: {max_sim:.4f}, Avg: {avg_sim:.4f}, Median: {median_sim:.4f}")
                                print(
                                    f"[DIARIZATION]   Matches: {match_count}/{total_samples} samples "
                                    f"above {self.wearer_multi_vote_sim_threshold:.2f}"
                                )
                                print(f"[DIARIZATION] ===========================================")
                                
                                # Decision logic: Adjusted for real-world similarity ranges
                                # Based on logs: same speaker typically gets 0.50-0.70 similarity
                                # Different speaker typically gets 0.20-0.45 similarity
                                
                                # Get similarities array from result
                                all_sims = np.array(result['all_similarities'])
                                strong_match_ratio = float(np.mean(all_sims >= self.wearer_multi_vote_sim_threshold)) if total_samples else 0.0
                                count_ratio = float(np.mean(all_sims >= self.wearer_full_count_sim_threshold)) if total_samples else 0.0

                                # Multiple criteria for detection (prevents false positives)
                                if avg_sim >= self.wearer_full_avg_threshold:
                                    is_wearer = True
                                    confidence_pct = avg_sim * 100
                                    print(f"[DIARIZATION] WEARER (AVG MATCH) - Avg: {confidence_pct:.1f}%, Median: {median_sim:.4f}")
                                
                                elif median_sim >= self.wearer_full_median_threshold and max_sim >= self.wearer_full_median_max_threshold:
                                    is_wearer = True
                                    print(f"[DIARIZATION] WEARER (MEDIAN+MAX) - Median: {median_sim:.4f}, Max: {max_sim:.4f}")
                                
                                elif strong_match_ratio >= self.wearer_full_match_ratio_threshold and max_sim >= self.wearer_full_match_ratio_max_threshold:
                                    is_wearer = True
                                    print(f"[DIARIZATION] WEARER (VOTE MATCH) - {match_count}/{total_samples} samples, Max: {max_sim:.4f}")
                                
                                elif count_ratio >= self.wearer_full_count_ratio_threshold:
                                    is_wearer = True
                                    matches_50 = int(np.sum(all_sims >= self.wearer_full_count_sim_threshold))
                                    print(
                                        f"[DIARIZATION] WEARER (MAJORITY) - {matches_50}/{total_samples} "
                                        f"above {self.wearer_full_count_sim_threshold:.2f}, Avg: {avg_sim:.4f}"
                                    )
                                
                                else:
                                    is_wearer = False
                                    print(f"[DIARIZATION] OTHER SPEAKER - Avg: {avg_sim:.4f}, Median: {median_sim:.4f}, Max: {max_sim:.4f}")
                                
                            elif registered_embedding_single is not None:
                                # Single-sample comparison (legacy/fallback)
                                print(f"[DIARIZATION] Using single-sample comparison")
                                similarity = self.compare_embeddings(segment_embedding, registered_embedding_single)
                                voice_similarity = similarity
                                
                                SIMILARITY_THRESHOLD = self.wearer_single_threshold_full
                                
                                print(f"[DIARIZATION] Similarity: {similarity:.4f} (threshold: {SIMILARITY_THRESHOLD})")
                                
                                if similarity >= SIMILARITY_THRESHOLD:
                                    is_wearer = True
                                    confidence_pct = similarity * 100
                                    print(f"[DIARIZATION] ✓ WEARER - Confidence: {confidence_pct:.1f}%")
                                else:
                                    print(f"[DIARIZATION] → OTHER SPEAKER - {similarity:.4f} < {SIMILARITY_THRESHOLD}")
                    
                    # Add segment with voice matching info
                    segment_data = {
                        'speaker_id': speaker,
                        'transcription': text,
                        'start': start_time,
                        'end': end_time,
                        'duration': duration,
                        'confidence': 0.9  # Default confidence for optimized processing
                    }
                    
                    # Add translation info if available
                    if transcription_result:
                        segment_data['original_text'] = transcription_result.get('original_text', text)
                        segment_data['translated'] = transcription_result.get('translated', False)
                        segment_data['source_lang'] = transcription_result.get('source_lang', transcription_lang or self.transcription_lang)
                        if transcription_result.get('target_lang'):
                            segment_data['target_lang'] = transcription_result.get('target_lang')
                    
                    # Add voice matching results if wearer's voice is registered
                    if has_registered_voice:
                        segment_data['is_wearer'] = is_wearer
                        segment_data['voice_similarity'] = voice_similarity
                    
                    segments.append(segment_data)
                    
                    segment_count += 1
                    if is_wearer:
                        print(f"[DIARIZATION] Added segment {segment_count}: {speaker} [WEARER] - '{text[:50]}{'...' if len(text) > 50 else ''}'")
                    else:
                        print(f"[DIARIZATION] Added segment {segment_count}: {speaker} [OTHER] - '{text[:50]}{'...' if len(text) > 50 else ''}'")
                
                # Only print detailed summary for full audio or when segments found
                if not is_chunk or len(segments) > 0:
                    print(f"[DIARIZATION] Processing complete: {len(segments)} valid segments found")
                    if len(segments) > 0:
                        print(f"[DIARIZATION] Filtering summary:")
                        print(f"  - Skipped short segments: {skipped_short}")
                        print(f"  - Skipped low energy: {skipped_energy}")
                        print(f"  - Skipped empty transcriptions: {skipped_empty}")
                        print(f"  - Final valid segments: {len(segments)}")
                # For empty chunks, completely silent - no logging at all
                
                processing_method = 'chunk_diarization' if is_chunk else 'optimized_diarization'
                return {
                    'segments': segments,
                    'total_duration': len(audio_array) / sample_rate,
                    'speaker_count': len(set(seg['speaker_id'] for seg in segments)),
                    'processing_method': processing_method,
                    'is_chunk': is_chunk
                }
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            
        except Exception as e:
            print(f"[DIARIZATION] Error processing audio array: {e}")
            return None

    def clear_cache(self):
        """Clear the audio cache to free memory."""
        with self.cache_lock:
            self.audio_cache.clear()
            print("[DIARIZATION] Audio cache cleared")

    def get_cache_stats(self):
        """Get cache statistics."""
        with self.cache_lock:
            return {
                'cache_size': len(self.audio_cache),
                'cache_keys': list(self.audio_cache.keys())[:10]  # First 10 keys
            }


