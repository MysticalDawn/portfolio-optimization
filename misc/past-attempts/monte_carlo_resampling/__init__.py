import sys
from pathlib import Path

# Add project root to path so 'misc' package can be found
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from monte_carlo_resampling import run

# =============================================================================
# RUNNER - Configure and run Monte Carlo resampling optimization
# =============================================================================

if __name__ == "__main__":
    run(
        shrinkage_intensity=0.7,
        period="11y",
        num_simulations=1000,
        num_target_returns=10,
    )
