# Community-Resolved Criticality in Polarization Dynamics

Numerical reproducibility repository for the manuscript:

**Community-Resolved Criticality in Polarization Dynamics: Mode Switching and Hidden Polarization in a Transverse-Field Mean-Field Model**

## Contents

- `numerical_validation.py` - complete numerical validation code used for the manuscript.
- `data/` - numerical branch and parameter-sweep data reported in the manuscript.
- `requirements.txt` - Python dependencies.

Running the script regenerates the seven validation figures in a local `figures/` directory and rewrites the numerical tables in `data/`.

## Numerical tests reproduced

The code verifies:

1. The cooperative critical threshold and near-critical extrapolation.
2. Square-root growth of the order parameter above onset.
3. Alignment of the nonlinear branch with the critical eigenvector.
4. Displacement of the critical threshold under community-dependent fluctuation strength.
5. Community-dominance mode switching.
6. Hidden polarization under antagonistic coupling.
7. The classical-limit scaling as the fluctuation parameters approach zero.

## Reproduction

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python numerical_validation.py
```

The script creates:

- `figures/fig1_threshold_extrapolation.{pdf,png}`
- `figures/fig2_square_root_scaling.{pdf,png}`
- `figures/fig3_eigenvector_alignment.{pdf,png}`
- `figures/fig4_gamma_threshold_shift.{pdf,png}`
- `figures/fig5_mode_switching.{pdf,png}`
- `figures/fig6_hidden_polarization.{pdf,png}`
- `figures/fig7_classical_limit.{pdf,png}`

and the CSV/JSON outputs stored under `data/`.

## Main benchmark values

For the cooperative benchmark, the code gives
`tau_c = 0.9362985683` and a fitted critical exponent `0.4995967`.

For the hidden-polarization benchmark, it gives
`tau_c = 0.7728818029`, while the global mean remains numerically zero along the antisymmetric branch.

## Citation

If you use this repository, please cite the associated manuscript once the final journal citation is available.

## Authors

Dorcas Attuabea Addo, Peter Akayuure, Thomas Mensah-Wonkyi, Peter Yeboah, Samuel Amoh Gyampoh, and Richard Kwame Ansah.
