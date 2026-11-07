"""vadkit: lightweight, modular, causal video anomaly detection for static cameras.

Two branches, fused per frame:
  * grid branch (the Arisoy / Kayabol statistical-background line): autoencoder residual scored per 16x16 cell with
    RX statistics fitted on normal video, Bayesian shrinkage of cell statistics towards the scene, optional Student-t;
  * object branch: light detector / pose / flow / embedding -> attributes scored by NIW mixtures.
Each channel is calibrated against its own normal reference (Fisher tail), channels are summed, then a causal median.
See vadkit/DESIGN.md.
"""
__version__ = "0.1.0"
