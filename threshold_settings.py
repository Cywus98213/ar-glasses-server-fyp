from __future__ import annotations

import os
from dataclasses import dataclass


def _getenv_float(name: str) -> float | None:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _getenv_int(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


@dataclass(frozen=True)
class ThresholdSettings:
    """
    Global, env-driven thresholds.

    Simple tuning (recommended):
      - SPEAKER_SIM_THRESHOLD: "same speaker" match threshold for speaker tracking
      - WEARER_SIM_THRESHOLD: threshold for matching against registered wearer embedding(s)

    Advanced tuning (optional): any of the specific WEARER_* / SPEAKER_* / DIAR_* variables.
    """

    # -------------- simple knobs (recommended) --------------
    speaker_sim_threshold: float = 0.32
    wearer_sim_threshold: float = 0.65

    # -------------- diarization tuning --------------
    diar_clustering_threshold: float = 0.5
    diar_min_cluster_size: int = 2
    diar_min_duration_off: float = 0.5

    # -------------- wearer detection (chunk + full audio) --------------
    wearer_multi_vote_sim_threshold: float = 0.62
    wearer_chunk_avg_threshold: float = 0.48
    wearer_chunk_median_threshold: float = 0.46
    wearer_chunk_max_threshold: float = 0.55
    wearer_chunk_match_ratio_threshold: float = 0.25
    wearer_chunk_match_ratio_max_threshold: float = 0.58
    wearer_chunk_count_sim_threshold: float = 0.48
    wearer_chunk_count_ratio_threshold: float = 0.35
    wearer_single_threshold_chunk: float = 0.65

    wearer_full_avg_threshold: float = 0.52
    wearer_full_median_threshold: float = 0.50
    wearer_full_median_max_threshold: float = 0.58
    wearer_full_match_ratio_threshold: float = 0.30
    wearer_full_match_ratio_max_threshold: float = 0.62
    wearer_full_count_sim_threshold: float = 0.50
    wearer_full_count_ratio_threshold: float = 0.40
    wearer_single_threshold_full: float = 0.70

    # -------------- speaker tracking / merging --------------
    speaker_match_threshold: float = 0.32
    speaker_merge_band_low: float = 0.28
    speaker_merge_band_high: float = 0.32
    speaker_proactive_merge_threshold: float = 0.35

    @staticmethod
    def from_env() -> "ThresholdSettings":
        # Start from defaults
        base = ThresholdSettings()

        # Simple knobs (if present) drive the rest unless overridden by advanced vars.
        speaker_simple = _getenv_float("SPEAKER_SIM_THRESHOLD")
        wearer_simple = _getenv_float("WEARER_SIM_THRESHOLD")

        speaker_sim = speaker_simple if speaker_simple is not None else base.speaker_sim_threshold
        wearer_sim = wearer_simple if wearer_simple is not None else base.wearer_sim_threshold

        # Advanced overrides (optional)
        diar_clustering_threshold = _getenv_float("DIAR_CLUSTERING_THRESHOLD")
        diar_min_cluster_size = _getenv_int("DIAR_MIN_CLUSTER_SIZE")
        diar_min_duration_off = _getenv_float("DIAR_MIN_DURATION_OFF")

        # If you only set WEARER_SIM_THRESHOLD, we map it to the most important thresholds.
        wearer_single_chunk = _getenv_float("WEARER_SINGLE_THRESHOLD_CHUNK")
        wearer_single_full = _getenv_float("WEARER_SINGLE_THRESHOLD_FULL")

        speaker_match = _getenv_float("SPEAKER_MATCH_THRESHOLD")
        speaker_merge_low = _getenv_float("SPEAKER_MERGE_BAND_LOW")
        speaker_merge_high = _getenv_float("SPEAKER_MERGE_BAND_HIGH")
        speaker_proactive_merge = _getenv_float("SPEAKER_PROACTIVE_MERGE_THRESHOLD")

        # Derive merge band from simple speaker threshold if not explicitly set.
        derived_merge_high = speaker_sim
        derived_merge_low = max(0.0, speaker_sim - 0.04)

        return ThresholdSettings(
            speaker_sim_threshold=speaker_sim,
            wearer_sim_threshold=wearer_sim,

            diar_clustering_threshold=diar_clustering_threshold if diar_clustering_threshold is not None else base.diar_clustering_threshold,
            diar_min_cluster_size=diar_min_cluster_size if diar_min_cluster_size is not None else base.diar_min_cluster_size,
            diar_min_duration_off=diar_min_duration_off if diar_min_duration_off is not None else base.diar_min_duration_off,

            # wearer: keep existing advanced defaults, but let the single simple knob shape the key ones
            wearer_multi_vote_sim_threshold=_getenv_float("WEARER_MULTI_VOTE_SIM_THRESHOLD") or base.wearer_multi_vote_sim_threshold,

            wearer_chunk_avg_threshold=_getenv_float("WEARER_CHUNK_AVG_THRESHOLD") or min(wearer_sim, base.wearer_chunk_avg_threshold),
            wearer_chunk_median_threshold=_getenv_float("WEARER_CHUNK_MEDIAN_THRESHOLD") or min(wearer_sim, base.wearer_chunk_median_threshold),
            wearer_chunk_max_threshold=_getenv_float("WEARER_CHUNK_MAX_THRESHOLD") or base.wearer_chunk_max_threshold,
            wearer_chunk_match_ratio_threshold=_getenv_float("WEARER_CHUNK_MATCH_RATIO_THRESHOLD") or base.wearer_chunk_match_ratio_threshold,
            wearer_chunk_match_ratio_max_threshold=_getenv_float("WEARER_CHUNK_MATCH_RATIO_MAX_THRESHOLD") or base.wearer_chunk_match_ratio_max_threshold,
            wearer_chunk_count_sim_threshold=_getenv_float("WEARER_CHUNK_COUNT_SIM_THRESHOLD") or min(wearer_sim, base.wearer_chunk_count_sim_threshold),
            wearer_chunk_count_ratio_threshold=_getenv_float("WEARER_CHUNK_COUNT_RATIO_THRESHOLD") or base.wearer_chunk_count_ratio_threshold,
            wearer_single_threshold_chunk=wearer_single_chunk if wearer_single_chunk is not None else wearer_sim,

            wearer_full_avg_threshold=_getenv_float("WEARER_FULL_AVG_THRESHOLD") or base.wearer_full_avg_threshold,
            wearer_full_median_threshold=_getenv_float("WEARER_FULL_MEDIAN_THRESHOLD") or base.wearer_full_median_threshold,
            wearer_full_median_max_threshold=_getenv_float("WEARER_FULL_MEDIAN_MAX_THRESHOLD") or base.wearer_full_median_max_threshold,
            wearer_full_match_ratio_threshold=_getenv_float("WEARER_FULL_MATCH_RATIO_THRESHOLD") or base.wearer_full_match_ratio_threshold,
            wearer_full_match_ratio_max_threshold=_getenv_float("WEARER_FULL_MATCH_RATIO_MAX_THRESHOLD") or base.wearer_full_match_ratio_max_threshold,
            wearer_full_count_sim_threshold=_getenv_float("WEARER_FULL_COUNT_SIM_THRESHOLD") or base.wearer_full_count_sim_threshold,
            wearer_full_count_ratio_threshold=_getenv_float("WEARER_FULL_COUNT_RATIO_THRESHOLD") or base.wearer_full_count_ratio_threshold,
            wearer_single_threshold_full=wearer_single_full if wearer_single_full is not None else max(wearer_sim, base.wearer_single_threshold_full),

            speaker_match_threshold=speaker_match if speaker_match is not None else speaker_sim,
            speaker_merge_band_low=speaker_merge_low if speaker_merge_low is not None else derived_merge_low,
            speaker_merge_band_high=speaker_merge_high if speaker_merge_high is not None else derived_merge_high,
            speaker_proactive_merge_threshold=speaker_proactive_merge if speaker_proactive_merge is not None else max(base.speaker_proactive_merge_threshold, speaker_sim + 0.03),
        )

