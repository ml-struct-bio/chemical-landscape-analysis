"""Values both the analysis and plotting layers need.

Anything here is by definition shared, so it must not import from either layer.
"""
from __future__ import annotations

# Plot-axis ranges for the two nuclei, matching the `shift_min`/`shift_max` the
# peak embedder itself normalizes with (nmr-to-3d/configs/config.yaml's
# feature_args for h_peak_centroid / c_peak_centroid). Peaks are handed to us in
# raw ppm -- the hcpeak condition applies an `identity` transform and the
# embedder normalizes internally -- so these are the model's own view of the
# usable shift window, not a cosmetic choice.
DEFAULT_H_PPM_RANGE = (-2.0, 12.0)
DEFAULT_C_PPM_RANGE = (-20.0, 230.0)
