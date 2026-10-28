"""Shared physical and normalization constants for the unified_reco pipeline."""

# LYSO calorimeter
NORM_POS_LYSO = 100.0   # mm
NORM_E_LYSO = 70.0      # MeV
NORM_T_LYSO = 500.0     # ns

# ATAR
NORM_POS_ATAR = 10.0    # mm
NORM_E_ATAR = 1.0       # MeV (dE/dx scale)
NORM_T_ATAR = 500.0     # ns

# Coincidence / physics windows
SIGMA_COINC_NS = 2.0    # Gaussian σ for LYSO-ATAR coincidence
TOF_NS = 0.5            # ATAR → LYSO mean time-of-flight approximation

# Acceptance criteria (pion stop + positron angle), raw units
ACCEPT_Z_MIN_MM = 1.2
ACCEPT_Z_MAX_MM = 4.8
ACCEPT_XY_MAX_MM = 8.0
ACCEPT_ANGLE_MAX_DEG = 120.0

# PDG bit flags (matches pileup_mixer.py)
PDG_PION     = 0b000001
PDG_MUON     = 0b000010
PDG_POSITRON = 0b000100
PDG_ELECTRON = 0b001000
PDG_GAMMA    = 0b010000
PDG_OTHER    = 0b100000
