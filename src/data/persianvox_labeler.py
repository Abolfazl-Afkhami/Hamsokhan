"""
Automated Soft Turn-Taking Labeler for PersianVox
Generates soft ramp targets [0.15, 0.35, 0.65, 1.0, 1.0, 1.0] and -100 ignore masks.
Distinguishes thinking pauses (< 720ms) from true conversational turn-ends.
"""

from typing import List, Dict
import torch


def build_persianvox_soft_labels(
    segments: List[Dict],
    total_frames_12_5hz: int,
    user_speaker_id: str = "spk_0",
    eot_silence_threshold_sec: float = 0.72
) -> torch.Tensor:
    """
    Args:
        segments: Sorted list of segment dicts: [{'speaker': str, 'start': float, 'end': float}, ...]
        total_frames_12_5hz: Total sequence length at 12.5Hz (80ms per frame).
        user_speaker_id: Identifier of the primary user speaker.
        eot_silence_threshold_sec: Threshold (0.72s = 9 frames) separating thinking pauses from turn completion.
    Returns:
        labels: [total_frames_12_5hz] tensor in FP16 with values {0.0, 0.15..1.0, -100.0}.
    """
    labels = torch.full((total_frames_12_5hz,), -100.0, dtype=torch.float32)
    ramp_values = torch.tensor([0.15, 0.35, 0.65, 1.0, 1.0, 1.0], dtype=torch.float32)

    for i, seg in enumerate(segments):
        if seg["speaker"] != user_speaker_id:
            continue

        s_frame = max(0, round(seg["start"] / 0.08))
        e_frame = min(total_frames_12_5hz - 1, round(seg["end"] / 0.08))

        # Default: user speech frames are 0.0 (turn ongoing)
        labels[s_frame : e_frame + 1] = 0.0

        # Determine if this segment ends the turn (EOT) or is an in-turn thinking pause
        is_eot = False
        if i == len(segments) - 1:
            is_eot = True
        else:
            next_seg = segments[i + 1]
            silence_gap = next_seg["start"] - seg["end"]
            if next_seg["speaker"] != user_speaker_id:
                is_eot = True  # Speaker switch
            elif silence_gap >= eot_silence_threshold_sec:
                is_eot = True  # True silence pause exceeding 720ms
            else:
                # In-turn thinking pause (< 720ms): silence frames are marked 0.0 (Hard Negative)
                next_s_frame = min(total_frames_12_5hz - 1, round(next_seg["start"] / 0.08))
                labels[e_frame : next_s_frame + 1] = 0.0

        if is_eot:
            # Apply soft ramp [0.15, 0.35, 0.65, 1.0, 1.0, 1.0] across [e_frame-3 ... e_frame+2]
            for offset, val in zip(range(-3, 3), ramp_values):
                idx = e_frame + offset
                if 0 <= idx < total_frames_12_5hz:
                    labels[idx] = val

    return labels.to(torch.float16)


if __name__ == "__main__":
    dummy_segments = [
        {"speaker": "spk_0", "start": 0.0, "end": 2.0},    # User speech
        {"speaker": "spk_0", "start": 2.3, "end": 4.0},    # In-turn pause (0.3s < 0.72s)
        {"speaker": "spk_1", "start": 4.5, "end": 6.0},    # Other speaker response (EOT at 4.0s)
    ]
    labels = build_persianvox_soft_labels(dummy_segments, total_frames_12_5hz=80)
    assert labels.shape == (80,)
    # Check that in-turn pause at ~2.1s (frame 26) is 0.0
    pause_frame = round(2.15 / 0.08)
    assert labels[pause_frame] == 0.0, f"Expected 0.0 for thinking pause, got {labels[pause_frame]}"
    print(f"✓ PersianVox soft labeler verified with hard negative pause modeling.")
