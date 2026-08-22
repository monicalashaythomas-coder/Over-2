"""
Probability calibration - spec section 14, DELIBERATELY SIMPLIFIED.

Full spec asks for isotonic regression / Platt scaling validated via
walk-forward. Per explicit instruction we are not building the
walk-forward engine today. What this module does instead, so that
"calibrated_probability" is never just an unvalidated number:

  1. Fit: given (raw_ensemble_probability, actual_outcome) pairs from
     a TRAIN split only, bucket raw probabilities into bins and record
     the actual win rate observed in each bin.
  2. Apply: at inference, map a new raw probability to the win rate of
     its bucket (linear interpolation between bucket centers).
  3. If a bucket has too few observations, it is not trusted and falls
     back to the raw probability (with confidence penalized) rather
     than reporting a confident number backed by ~3 samples.

This is a single train/test split, not full walk-forward - it will
NOT catch calibration drift over time or regime change. It exists so
the pipeline is never emitting "calibrated" numbers with literally
zero out-of-sample check behind them. Treat any live results with
that limitation in mind.
"""
from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import List, Tuple

MIN_BUCKET_SAMPLES = 25


@dataclass
class Bucket:
    low: float
    high: float
    center: float
    n: int
    actual_win_rate: float


class Calibrator:
    def __init__(self, n_buckets: int = 10):
        self.n_buckets = n_buckets
        self.buckets: List[Bucket] = []
        self.fitted = False

    def fit(self, pairs: List[Tuple[float, bool]]) -> None:
        """pairs: list of (raw_probability, outcome_win)."""
        clean = [(p, w) for p, w in pairs if not math.isnan(p)]
        if len(clean) < MIN_BUCKET_SAMPLES:
            self.fitted = False
            return

        edges = [i / self.n_buckets for i in range(self.n_buckets + 1)]
        self.buckets = []
        for i in range(self.n_buckets):
            low, high = edges[i], edges[i + 1]
            in_bucket = [w for p, w in clean if (low <= p < high) or (i == self.n_buckets - 1 and p == high)]
            n = len(in_bucket)
            if n == 0:
                continue
            actual = sum(in_bucket) / n
            self.buckets.append(Bucket(low=low, high=high, center=(low + high) / 2, n=n, actual_win_rate=actual))
        self.fitted = len(self.buckets) > 0

    def calibrate(self, raw_p: float) -> Tuple[float, bool]:
        """Returns (calibrated_probability, trusted). trusted=False means
        the relevant bucket had too few samples and raw_p was returned
        unmodified - callers should treat this as lower-confidence."""
        if math.isnan(raw_p) or not self.fitted:
            return raw_p, False

        centers = [b.center for b in self.buckets]
        idx = bisect.bisect_left(centers, raw_p)

        if idx == 0:
            bucket = self.buckets[0]
        elif idx >= len(self.buckets):
            bucket = self.buckets[-1]
        else:
            left, right = self.buckets[idx - 1], self.buckets[idx]
            # linear interpolation between the two nearest bucket centers
            span = right.center - left.center
            t = 0.0 if span == 0 else (raw_p - left.center) / span
            interp = left.actual_win_rate + t * (right.actual_win_rate - left.actual_win_rate)
            trusted = left.n >= MIN_BUCKET_SAMPLES and right.n >= MIN_BUCKET_SAMPLES
            return interp, trusted

        return bucket.actual_win_rate, bucket.n >= MIN_BUCKET_SAMPLES

    def report(self) -> List[Bucket]:
        return self.buckets
