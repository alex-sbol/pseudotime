"""Period detection utilities."""
from __future__ import annotations

import numpy as np
from scipy import signal

from .config import PeriodConfig
from .data_models import CorridorSignals, PeriodEstimate


def autocorrelation_period(width: np.ndarray, config: PeriodConfig) -> float | None:
    """Estimate period via autocorrelation peaks within configured lag bounds."""
    # Remove DC component to focus on repeating structure.
    norm = width - np.mean(width)
    corr = signal.correlate(norm, norm, mode="full")
    lags = signal.correlation_lags(norm.size, norm.size, mode="full")
    pos_mask = lags > 0
    corr = corr[pos_mask]
    lags = lags[pos_mask]
    valid = (lags >= config.min_period_pixels) & (lags <= config.max_period_pixels)
    if not np.any(valid):
        return None
    # Find prominent peaks within allowable lag range.
    peaks, properties = signal.find_peaks(corr[valid], prominence=config.prominence * np.max(corr[valid]))
    if len(peaks) == 0:
        return None
    best_idx = peaks[np.argmax(properties["prominences"])]
    return float(lags[valid][best_idx])


def fft_period(width: np.ndarray, config: PeriodConfig) -> float | None:
    """Estimate period from the dominant FFT frequency component."""
    # Remove mean before FFT to suppress zero-frequency energy.
    norm = width - np.mean(width)
    fft = np.fft.rfft(norm)
    freqs = np.fft.rfftfreq(norm.size, d=1.0)
    freqs = freqs[1:]
    fft = fft[1:]
    if len(freqs) == 0:
        return None
    # Convert frequency axis back to spatial period and keep those within bounds.
    mask = (1 / freqs >= config.min_period_pixels) & (1 / freqs <= config.max_period_pixels)
    if not np.any(mask):
        return None
    best = np.argmax(np.abs(fft[mask]))
    best_freq = freqs[mask][best]
    if best_freq == 0:
        return None
    return float(1 / best_freq)


def estimate_period(signals: CorridorSignals, config: PeriodConfig) -> PeriodEstimate:
    """Combine autocorrelation and FFT heuristics into a robust period estimate."""
    # Combine both estimators and fall back if one failed.
    ac_period = autocorrelation_period(signals.corridor_width, config)
    fft_period_est = fft_period(signals.corridor_width, config)

    candidates = [p for p in (ac_period, fft_period_est) if p is not None]
    if not candidates:
        raise RuntimeError("Could not determine period from signals")

    period = float(np.median(candidates))
    # Approximate phase by locating the narrowest corridor section.
    phase_offset = float(np.argmin(signals.corridor_width))

    method = "+".join([m for m, p in (("ac", ac_period), ("fft", fft_period_est)) if p is not None])
    return PeriodEstimate(period=period, phase_offset=phase_offset, method=method)
