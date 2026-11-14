# Community-Resolved Criticality in Polarization Dynamics

Numerical reproducibility repository for the manuscript:

**Community-Resolved Criticality in Polarization Dynamics: Community-Dominance Crossover and Hidden Polarization in a Transverse-Field Mean-Field Model**

## Contents

- `numerical_validation.py` - complete numerical validation code used for the manuscript.
- `data/` - numerical branch and parameter-sweep data reported in the manuscript.
- `revision_numerical_additions.py` - additional continuation, asymmetry-robustness, and four-community analyses added in the referee revision.
- `requirements.txt` - Python dependencies.

Running `numerical_validation.py` regenerates Figures 1-7. Running `revision_numerical_additions.py` regenerates Figures 8-11 and the additional revision data tables.

## Numerical tests reproduced

The code verifies:

1. The cooperative critical threshold and near-critical extrapolation.
2. Square-root growth of the order parameter above onset.
3. Alignment of the nonlinear branch with the critical eigenvector.
4. Displacement of the critical threshold under community-dependent fluctuation strength.
5. Continuous community-dominance crossover of the critical eigenvector.
6. Hidden polarization under antagonistic coupling.
7. The classical-limit scaling as the fluctuation parameters approach zero.
8. Community-resolved equilibria and stability over a broad interaction range.
9. Robustness of hidden oppositional ordering under small weight/response asymmetries.
10. A four-community continuation example.

## Reproduction

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python numerical_validation.py
python revision_numerical_additions.py
```

The script creates:

- `figures/fig1_threshold_extrapolation.{pdf,png}`
- `figures/fig2_square_root_scaling.{pdf,png}`
- `figures/fig3_eigenvector_alignment.{pdf,png}`
- `figures/fig4_gamma_threshold_shift.{pdf,png}`
- `figures/fig5_mode_switching.{pdf,png}`
- `figures/fig6_hidden_polarization.{pdf,png}`
- `figures/fig7_classical_limit.{pdf,png}`
- `figures/fig8_community_branches.{pdf,png}`
- `figures/fig9_branch_stability.{pdf,png}`
- `figures/fig10_hidden_asymmetry_robustness.{pdf,png}`
- `figures/fig11_four_community_branch.{pdf,png}`

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
