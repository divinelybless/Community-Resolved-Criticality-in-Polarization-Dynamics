import json
from pathlib import Path
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig
from scipy.optimize import root, brentq

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'figures'
DATA = ROOT / 'data'
OUT.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'legend.fontsize': 9,
    'figure.figsize': (6.4, 4.2),
    'savefig.bbox': 'tight',
})

def q_nu(beta, gamma):
    gamma = np.asarray(gamma, dtype=float)
    q = np.empty_like(gamma)
    nu = np.empty_like(gamma)
    small = np.abs(gamma) < 1e-12
    q[small] = beta
    nu[small] = -(beta**3) / 3.0
    if np.any(~small):
        g = gamma[~small]
        z = beta * g
        q[~small] = np.tanh(z) / g
        nu[~small] = (beta * g / np.cosh(z)**2 - np.tanh(z)) / (2.0 * g**3)
    return q, nu

def response(x, beta, gamma):
    x = np.asarray(x, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    R = np.sqrt(x*x + gamma*gamma)
    y = np.zeros_like(x)
    nz = R > 1e-14
    y[nz] = x[nz] / R[nz] * np.tanh(beta * R[nz])
    return y

def response_prime(x, beta, gamma):
    x = np.asarray(x, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    R = np.sqrt(x*x + gamma*gamma)
    gp = np.empty_like(x)
    nz = R > 1e-12
    if np.any(nz):
        rn = R[nz]
        xn = x[nz]
        gn = gamma[nz]
        z = beta * rn
        gp[nz] = (gn**2 / rn**3) * np.tanh(z) + beta * (xn**2 / rn**2) / np.cosh(z)**2
    if np.any(~nz):
        gp[~nz] = beta
    return gp

def fp_residual(m, tau, beta, alpha, gamma, J):
    x = tau * (J @ (alpha * m))
    return m - response(x, beta, gamma)

def linear_data(beta, alpha, gamma, J):
    q, nu = q_nu(beta, gamma)
    M = np.diag(q) @ J @ np.diag(alpha)
    vals, vr = eig(M)
    idx = int(np.argmax(vals.real))
    lam = float(vals[idx].real)
    v = vr[:, idx].real
    if np.sum(v) < 0:
        v = -v
    v /= np.linalg.norm(v)
    vals_l, vl = eig(M.T)
    idx_l = int(np.argmin(np.abs(vals_l.real - lam)))
    w = vl[:, idx_l].real
    if np.dot(w, v) < 0:
        w = -w
    w /= np.dot(w, v)
    tc = 1.0 / lam
    K = J @ np.diag(alpha)
    kappa = -(tc**3) * np.sum(w * nu * (K @ v)**3)
    return q, nu, M, lam, tc, v, w, float(kappa)

def solve_branch(tau, beta, alpha, gamma, J, init=None):
    q, nu, M, lam, tc, v, w, kappa = linear_data(beta, alpha, gamma, J)
    mu = tau / tc - 1.0
    if init is None:
        amp = math.sqrt(max(mu, 1e-12) / kappa)
        init = max(amp, 1e-5) * v
    sol = root(fp_residual, init, args=(tau, beta, alpha, gamma, J), method='hybr', tol=1e-12)
    if np.linalg.norm(sol.fun) > 1e-9:
        raise RuntimeError(f'Root solve failed: residual={np.linalg.norm(sol.fun)}')
    m = sol.x
    if np.dot(m, v) < 0:
        m = -m
    return m

def branch_stability(m, tau, beta, alpha, gamma, J):
    x = tau * (J @ (alpha * m))
    gp = response_prime(x, beta, gamma)
    jac = -np.eye(len(m)) + tau * np.diag(gp) @ J @ np.diag(alpha)
    return float(np.max(np.linalg.eigvals(jac).real))

def savefig(fig, stem):
    fig.savefig(OUT / f'{stem}.pdf')
    fig.savefig(OUT / f'{stem}.png', dpi=300)
    plt.close(fig)

beta = 1.30
alpha = np.array([0.55, 0.45])
gamma = np.array([0.55, 1.00])
J = np.array([[1.40, 0.75], [0.75, 1.10]])
q, nu, M, lam, tc, vc, wc, kappa = linear_data(beta, alpha, gamma, J)

mus = np.logspace(-5, -1, 70)
taus = tc * (1.0 + mus)
branch = np.array([solve_branch(t, beta, alpha, gamma, J) for t in taus])
amps = np.linalg.norm(branch, axis=1)
angles, stability = [], []
for m, t in zip(branch, taus):
    u = m / np.linalg.norm(m)
    cosang = np.clip(abs(np.dot(u, vc)), 0.0, 1.0)
    angles.append(np.degrees(np.arccos(cosang)))
    stability.append(branch_stability(m, t, beta, alpha, gamma, J))
angles = np.array(angles)
stability = np.array(stability)

fit_threshold_mask = mus <= 5e-3
p_tc = np.polyfit(taus[fit_threshold_mask], amps[fit_threshold_mask]**2, 1)
tc_fit = -p_tc[1] / p_tc[0]
fit_exp_mask = mus <= 5e-3
crit_exp, log_pref = np.polyfit(np.log(mus[fit_exp_mask]), np.log(amps[fit_exp_mask]), 1)
pref_fit = float(np.exp(log_pref))
pref_theory = 1.0 / math.sqrt(kappa)

fig, ax = plt.subplots()
ax.plot(taus[fit_threshold_mask], amps[fit_threshold_mask]**2, 'o', label='Numerical branch')
tau_line = np.linspace(tc_fit, taus[fit_threshold_mask][-1], 200)
ax.plot(tau_line, np.polyval(p_tc, tau_line), '--', label='Linear fit')
ax.axvline(tc, linestyle=':', label=r'Analytic $\tau_c$')
ax.set_xlabel(r'$\tau$'); ax.set_ylabel(r'$\|\mathbf{m}\|_2^2$')
ax.set_title('Near-critical threshold extrapolation'); ax.legend()
savefig(fig, 'fig1_threshold_extrapolation')

fig, ax = plt.subplots()
ax.loglog(mus, amps, 'o', markersize=3, label='Numerical branch')
ax.loglog(mus, np.sqrt(mus / kappa), '--', label=r'Theory $\sqrt{\mu/\kappa}$')
ax.loglog(mus, pref_fit * mus**crit_exp, ':', label=fr'Fit $\mu^{{{crit_exp:.4f}}}$')
ax.set_xlabel(r'$\mu=\tau/\tau_c-1$'); ax.set_ylabel(r'$\|\mathbf{m}\|_2$')
ax.set_title('Square-root growth above the critical point'); ax.legend()
savefig(fig, 'fig2_square_root_scaling')

fig, ax = plt.subplots()
ax.loglog(mus, angles, 'o', markersize=3)
ax.set_xlabel(r'$\mu=\tau/\tau_c-1$'); ax.set_ylabel('Angle to critical eigenvector (degrees)')
ax.set_title('Alignment of the nonlinear branch with the critical mode')
savefig(fig, 'fig3_eigenvector_alignment')

gamma1_grid = np.linspace(0.0, 2.0, 161)
tc_curve = np.array([linear_data(beta, alpha, np.array([g1, gamma[1]]), J)[4] for g1 in gamma1_grid])
gamma1_samples = np.linspace(0.0, 2.0, 9)
tc_empirical, tc_exact_samples = [], []
for g1 in gamma1_samples:
    gam = np.array([g1, gamma[1]])
    _, _, _, _, tci, vi, _, kappai = linear_data(beta, alpha, gam, J)
    local_mu = np.geomspace(1e-4, 5e-3, 12)
    local_tau = tci * (1.0 + local_mu)
    local_amp2 = []
    for mu_i, t_i in zip(local_mu, local_tau):
        mi = solve_branch(t_i, beta, alpha, gam, J, init=math.sqrt(mu_i/kappai)*vi)
        local_amp2.append(np.dot(mi, mi))
    coeff = np.polyfit(local_tau, np.array(local_amp2), 1)
    tc_empirical.append(-coeff[1] / coeff[0]); tc_exact_samples.append(tci)
tc_empirical = np.array(tc_empirical); tc_exact_samples = np.array(tc_exact_samples)
gamma_threshold_max_rel_error = float(np.max(np.abs((tc_empirical - tc_exact_samples)/tc_exact_samples)))

fig, ax = plt.subplots()
ax.plot(gamma1_grid, tc_curve, label=r'Analytic $\tau_c(\Gamma_1)$')
ax.plot(gamma1_samples, tc_empirical, 'o', label='Continuation estimate')
ax.set_xlabel(r'$\Gamma_1$ (with $\Gamma_2=1$)'); ax.set_ylabel(r'$\tau_c$')
ax.set_title('Fluctuation-induced displacement of the critical threshold'); ax.legend()
savefig(fig, 'fig4_gamma_threshold_shift')

R1 = J[0,0]*alpha[0] + J[0,1]*alpha[1]
R2 = J[1,0]*alpha[0] + J[1,1]*alpha[1]
q2 = q_nu(beta, np.array([gamma[1]]))[0][0]
target_q1 = q2 * R2 / R1
switch_gamma1 = brentq(lambda g1: np.tanh(beta*g1)/g1 - target_q1, 1.0, 2.0)
gamma1_switch_grid = np.linspace(0.2, 2.2, 101)
r_eig, r_num = [], []
for g1 in gamma1_switch_grid:
    gam = np.array([g1, gamma[1]])
    _, _, _, _, tci, vi, _, kappai = linear_data(beta, alpha, gam, J)
    if vi[0] < 0: vi = -vi
    r_eig.append(vi[1] / vi[0])
    mu_probe = 1e-3
    mi = solve_branch(tci*(1+mu_probe), beta, alpha, gam, J, init=math.sqrt(mu_probe/kappai)*vi)
    if mi[0] < 0: mi = -mi
    r_num.append(mi[1] / mi[0])
r_eig = np.array(r_eig); r_num = np.array(r_num)
mode_ratio_max_abs_error = float(np.max(np.abs(r_num-r_eig)))

def nonlinear_mode_ratio(g1):
    gam = np.array([g1, gamma[1]])
    _, _, _, _, tci, vi, _, kappai = linear_data(beta, alpha, gam, J)
    if vi[0] < 0: vi = -vi
    mu_probe = 1e-3
    mi = solve_branch(tci*(1+mu_probe), beta, alpha, gam, J, init=math.sqrt(mu_probe/kappai)*vi)
    if mi[0] < 0: mi = -mi
    return mi[1]/mi[0]

switch_gamma1_numerical = brentq(lambda g1: nonlinear_mode_ratio(g1)-1.0, 1.2, 1.45)

fig, ax = plt.subplots()
ax.plot(gamma1_switch_grid, r_eig, label='Critical eigenvector ratio')
ax.plot(gamma1_switch_grid[::5], r_num[::5], 'o', label=r'Nonlinear ratio at $\mu=10^{-3}$')
ax.axhline(1.0, linestyle=':'); ax.axvline(switch_gamma1, linestyle='--', label=fr'Switch $\Gamma_1={switch_gamma1:.4f}$')
ax.set_xlabel(r'$\Gamma_1$ (with $\Gamma_2=1$)'); ax.set_ylabel(r'$m_2/m_1$ near onset')
ax.set_title('Community-dominance switching of the critical mode'); ax.legend()
savefig(fig, 'fig5_mode_switching')

beta_h = 1.40
alpha_h = np.array([0.5, 0.5])
gamma_h = np.array([0.55, 0.55])
J_h = np.array([[1.0, -1.2], [-1.2, 1.0]])
q_h, nu_h, M_h, lam_h, tc_h, vh, wh, kappa_h = linear_data(beta_h, alpha_h, gamma_h, J_h)
if vh[0] < 0: vh = -vh
mu_h = np.logspace(-5, math.log10(0.5), 90)
tau_h = tc_h * (1.0 + mu_h)
branch_h = []
for mu_i, t_i in zip(mu_h, tau_h):
    mi = solve_branch(t_i, beta_h, alpha_h, gamma_h, J_h, init=math.sqrt(mu_i/kappa_h)*vh)
    if mi[0] < 0: mi = -mi
    branch_h.append(mi)
branch_h = np.array(branch_h)
Mglob = branch_h @ alpha_h
Pcom = np.sqrt(np.sum(alpha_h[None,:]*branch_h**2, axis=1))
mask_h = mu_h <= 5e-3
coef_h = np.polyfit(tau_h[mask_h], Pcom[mask_h]**2, 1)
tc_h_fit = -coef_h[1] / coef_h[0]
max_hidden_global = float(np.max(np.abs(Mglob)))

fig, ax = plt.subplots()
ax.plot(tau_h/tc_h, Pcom, label=r'$P_{\rm com}$')
ax.plot(tau_h/tc_h, np.abs(Mglob), '--', label=r'$|M_{\rm glob}|$')
ax.axvline(1.0, linestyle=':', label=r'$\tau=\tau_c$')
ax.set_xlabel(r'$\tau/\tau_c$'); ax.set_ylabel('Order parameter')
ax.set_title('Hidden polarization under antagonistic coupling'); ax.legend()
savefig(fig, 'fig6_hidden_polarization')

base_gamma = gamma.copy()
scales = np.logspace(-3, 0, 70)
tc_scale = np.array([linear_data(beta, alpha, s*base_gamma, J)[4] for s in scales])
tc_classical = linear_data(beta, alpha, np.zeros_like(base_gamma), J)[4]
relative_shift = (tc_scale - tc_classical)/tc_classical
classical_fit_mask = scales <= 0.1
classical_exp, classical_logc = np.polyfit(np.log(scales[classical_fit_mask]), np.log(relative_shift[classical_fit_mask]), 1)
classical_pref = float(np.exp(classical_logc))

fig, ax = plt.subplots()
ax.loglog(scales, relative_shift, 'o', markersize=3, label='Threshold shift')
ax.loglog(scales, classical_pref*scales**classical_exp, '--', label=fr'Fit $s^{{{classical_exp:.4f}}}$')
ax.set_xlabel(r'Common fluctuation scale $s$, $\Gamma_a\mapsto s\Gamma_a$')
ax.set_ylabel(r'$(\tau_c(s)-\tau_c(0))/\tau_c(0)$')
ax.set_title('Convergence to the classical mean-field limit'); ax.legend()
savefig(fig, 'fig7_classical_limit')

summary = {
    'cooperative': {'q': q.tolist(), 'lambda_max': lam, 'tau_c_analytic': tc,
        'tau_c_continuation_fit': float(tc_fit), 'tau_c_relative_error': float((tc_fit-tc)/tc),
        'critical_eigenvector': vc.tolist(), 'kappa': kappa, 'critical_exponent_fit': float(crit_exp),
        'amplitude_prefactor_fit': pref_fit, 'amplitude_prefactor_theory': pref_theory,
        'angle_deg_at_smallest_mu': float(angles[0]),
        'max_real_jacobian_on_computed_branch': float(np.max(stability))},
    'gamma_sweep': {'max_relative_threshold_error': gamma_threshold_max_rel_error,
        'samples_gamma1': gamma1_samples.tolist(), 'analytic_tau_c': tc_exact_samples.tolist(),
        'continuation_tau_c': tc_empirical.tolist()},
    'mode_switching': {'R1': float(R1), 'R2': float(R2), 'gamma1_switch_analytic': float(switch_gamma1),
        'gamma1_switch_numerical_mu_1e-3': float(switch_gamma1_numerical),
        'gamma1_switch_relative_error': float((switch_gamma1_numerical-switch_gamma1)/switch_gamma1),
        'max_abs_ratio_error_at_mu_1e-3': mode_ratio_max_abs_error},
    'hidden_polarization': {'q': float(q_h[0]), 'lambda_max': lam_h, 'tau_c_analytic': tc_h,
        'tau_c_continuation_fit': float(tc_h_fit), 'tau_c_relative_error': float((tc_h_fit-tc_h)/tc_h),
        'kappa': kappa_h, 'max_abs_global_mean': max_hidden_global, 'max_Pcom_in_plot': float(np.max(Pcom))},
    'classical_limit': {'tau_c_classical': tc_classical, 'quadratic_exponent_fit': float(classical_exp)}
}

with open(DATA / 'epjp_numerical_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

np.savetxt(DATA / 'epjp_gamma_threshold_sweep.csv',
           np.column_stack([gamma1_samples, tc_exact_samples, tc_empirical]),
           delimiter=',', header='Gamma1,tau_c_analytic,tau_c_continuation', comments='')
np.savetxt(DATA / 'epjp_mode_switching.csv',
           np.column_stack([gamma1_switch_grid, r_eig, r_num]),
           delimiter=',', header='Gamma1,eigenvector_ratio,numerical_ratio_mu_1e-3', comments='')
np.savetxt(DATA / 'epjp_cooperative_branch.csv',
           np.column_stack([mus, taus, branch[:,0], branch[:,1], amps, angles, stability]),
           delimiter=',', header='mu,tau,m1,m2,norm_m,angle_deg,max_real_jacobian', comments='')
np.savetxt(DATA / 'epjp_hidden_branch.csv',
           np.column_stack([mu_h, tau_h, branch_h[:,0], branch_h[:,1], Mglob, Pcom]),
           delimiter=',', header='mu,tau,m1,m2,Mglob,Pcom', comments='')

print(json.dumps(summary, indent=2))
