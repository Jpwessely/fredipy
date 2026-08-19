"""Coverage for ``GaussLegendre_1D_log``'s analytic UV tail correction.

The correction implements

    Sigma_12 = BB + A_1 T_2^T + T_1 A_2^T + T_1 T_2^T          (**)

for a pair of integral constraints, where BB is the plain double quadrature, T_i are the
per-constraint, per-row analytic tail moments above the UV split point w_uv, and A_i are the
bulk-UV overlaps evaluated through the anchor K(w, w_uv) / f_uv(w_uv).

The tests below pin down, in order: that (**) makes the cross block consistent with the
prediction operator (T1), that the assembled covariance stays symmetric (T2), that the legacy
symmetric closed form is unchanged (T3), that the sum-rule tail moment is the right analytic
value (T4), that the end-to-end posterior is self-consistent (T5), that plain quadrature models
are bit-identical to their pre-fix values (T6), and the two error/shape guards (T7, T8).
"""

import numpy as np
import pytest

from fredipy import constraints, integrators, kernels, models, operators
from fredipy.covariance import OneSided, TwoSided
from fredipy.util import make_column_vector, make_row_vector


# --------------------------------------------------------------------------------------------
# Synthetic, deliberately well-conditioned model.
#
# Tolerances in an end-to-end test are limited by cond(OpKerOp + cov_y); the production gluon and
# ghost models run at data_cov = 1e-10 where cond ~ 7e12 and no identity can be checked below
# ~1e-5. These settings keep the model far from that regime so the tests measure the correction,
# not the conditioning.
# --------------------------------------------------------------------------------------------

W_MIN = 1e-3          # Lower end of the omega quadrature range.
W_UV = 1e5            # Upper end of the quadrature range == UV split point w_uv.
INT_N = 120           # Number of Gauss-Legendre nodes on (W_MIN, W_UV).
DATA_COV = 1e-6       # Diagonal data covariance; large enough to keep the model well conditioned.
SR_COV = 1e-8         # Sum-rule target covariance.
N_DATA = 15           # Number of synthetic momentum points.
P_MIN, P_MAX = 0.5, 20.0   # Momentum window of the synthetic data, in the same units as omega.

RBF_VARIANCE = 2.0    # Base RBF variance of the synthetic kernel.
RBF_LENGTHSCALE = 1.5  # Base RBF lengthscale of the synthetic kernel.
MU_UV = 1.1           # Centre of the UV blend-in; must satisfy W_UV >> MU_UV for (*) to hold.
L_UV = 0.15           # Width of the UV blend-in.

GAMMA_GLUON = 13.0 / 22.0  # Gluon anomalous dimension (the exponent formerly hardcoded).
GAMMA_GHOST = 9.0 / 44.0   # Ghost anomalous dimension; differs, which is why it is a parameter.


def propagator_uv_shape(w, gamma=GAMMA_GLUON):
    """Propagator-type UV asymptotic shape f_uv(w) = w^-2 (2 log w)^-(1+gamma).

    Uses the same smooth-maximum regularisation as the reconstructions repo's ``uv_asymptotics``
    so that the log stays finite below w = 1; the regularisation is irrelevant at w >= w_uv.
    """
    delta = 0.05  # Smoothing width of the regularised maximum.
    w1 = 0.5 * (w + 1.15 + np.sqrt((w - 1.15) ** 2 + delta ** 2))
    return 1.0 / (w1 ** 2 * np.log(w1 ** 2) ** (1.0 + gamma))


def ghost_uv_shape(w):
    """Propagator-type UV shape at the ghost anomalous dimension."""
    return propagator_uv_shape(w, gamma=GAMMA_GHOST)


def kl_p2(p, w):
    """Kallen-Lehmann data kernel multiplied by p^2."""
    return p ** 2 * w / (w ** 2 + p ** 2) / np.pi


def sr_kernel(p, w):
    """Sum-rule kernel C(p, w) = w, broadcast over the (ignored) momentum axis."""
    return w * np.ones_like(p)


def analytic_sum_rule_tail(w_uv, gamma):
    """Closed form (2 log w_uv)^-gamma / (2 gamma) of the sum-rule tail moment."""
    return (2.0 * np.log(w_uv)) ** (-gamma) / (2.0 * gamma)


def sum_rule_tail_moment(w_uv, gamma):
    """Row-independent ``tail_moment`` callable for the sum-rule kernel C(p, w) = w."""
    value = analytic_sum_rule_tail(w_uv, gamma)

    def _tail_moment(x):
        return np.full((np.atleast_2d(x).shape[0], 1), value)

    return _tail_moment


def make_kernel(uv_shape=propagator_uv_shape):
    """AsymptoticKernel with a UV branch only (no IR branch) at the synthetic hyperparameters."""
    rbf = kernels.RadialBasisFunction(RBF_VARIANCE, RBF_LENGTHSCALE)
    kernel = kernels.AsymptoticKernel(rbf)
    kernel.add_asymptotics(region="UV", asymptotics=uv_shape)
    kernel.set_params(asymp_params=[MU_UV, L_UV])
    return kernel


def synthetic_data():
    """Momenta and correlator values of the synthetic data set."""
    p = np.geomspace(P_MIN, P_MAX, N_DATA)
    return p, p ** 2 / (p ** 2 + 1.0)


def build_model(with_tail=True, gamma=GAMMA_GLUON, uv_shape=propagator_uv_shape):
    """Two-constraint model: a KL data constraint plus a one-row sum rule.

    ``with_tail=False`` gives both constraints a plain ``GaussLegendre_1D_log`` on the same grid,
    which is what the tgvert/ghgvert/fgvertsd projects do and what T6 pins down.
    """
    p, G = synthetic_data()
    kernel = make_kernel(uv_shape)

    data_integrator = integrators.GaussLegendre_1D_log(W_MIN, W_UV, INT_N)
    c_data = constraints.LinearEquality(
        operators.Integral(kl_p2, data_integrator),
        {"x": p, "y": G, "cov_y": DATA_COV * np.ones_like(G)},
    )

    if with_tail:
        sr_integrator = integrators.GaussLegendre_1D_log_UVtail(
            W_MIN, W_UV, INT_N, uv_shape, tail_moment=sum_rule_tail_moment(W_UV, gamma)
        )
    else:
        sr_integrator = integrators.GaussLegendre_1D_log(W_MIN, W_UV, INT_N)
    c_sr = constraints.LinearEquality(
        operators.Integral(sr_kernel, sr_integrator),
        {"x": np.array([0.0]), "y": np.array([0.0]), "cov_y": np.array([SR_COV])},
    )

    model = models.GaussianProcess(kernel, [c_data, c_sr])
    return model, c_data, c_sr, data_integrator, sr_integrator


def bulk_double_integration(integrator, c1, kernel, c2):
    """Uncorrected double quadrature BB, written out so the tests do not lean on the fix."""
    row1 = integrator.weights * c1(make_row_vector(integrator.w), x=make_column_vector(c1.x))
    row2 = integrator.weights * c2(make_row_vector(integrator.w), x=make_column_vector(c2.x))
    return row1 @ kernel(integrator.w, integrator.w) @ row2.T


# --------------------------------------------------------------------------------------------
# T1 -- the sharpest test: pure linear algebra, no linear solve, so it must hold to machine
# precision. The cross block of OpKerOp is, by definition, the data-side reintegration of the
# sum-rule row of the prediction operator OpKer. That identity is exactly what the missing
# cross-block correction used to break.
# --------------------------------------------------------------------------------------------

def test_cross_block_equals_reintegrated_opker_row():
    model, c_data, c_sr, data_integrator, _ = build_model()
    kernel = model.kernel
    n_data = c_data.x.shape[0]

    opkerop = TwoSided()(kernel, [c_data, c_sr])
    cross_block = opkerop[:n_data, n_data:]                       # (M, 1)

    nodes = data_integrator.w
    opker = OneSided()(kernel, [c_data, c_sr], nodes)             # (M + 1, N)
    sr_row = opker[n_data:, :]                                    # (1, N)

    data_weighted = data_integrator.weights * kl_p2(
        make_column_vector(c_data.x), make_row_vector(nodes)
    )
    reintegrated = data_weighted @ sr_row.T                       # (M, 1)

    np.testing.assert_allclose(cross_block, reintegrated, rtol=1e-12)


# --------------------------------------------------------------------------------------------
# T2 -- np.linalg.cholesky in models.py reads only the lower triangle, so an asymmetric OpKerOp
# would never raise. This is the only guard against that.
# --------------------------------------------------------------------------------------------

def test_covariance_matrix_is_symmetric():
    model, c_data, c_sr, _, _ = build_model()
    kernel = model.kernel
    n_data = c_data.x.shape[0]

    opkerop = TwoSided()(kernel, [c_data, c_sr])
    scale = np.abs(opkerop).max()
    np.testing.assert_allclose(opkerop, opkerop.T, rtol=1e-14, atol=1e-14 * scale)

    # Building the same model with the constraint order reversed must give the permuted matrix,
    # i.e. block (sr, data) must carry exactly the same correction as block (data, sr).
    reversed_opkerop = TwoSided()(kernel, [c_sr, c_data])
    perm = np.concatenate([np.arange(n_data, n_data + 1), np.arange(n_data)])
    np.testing.assert_allclose(
        reversed_opkerop, opkerop[np.ix_(perm, perm)], rtol=1e-14, atol=1e-14 * scale
    )


# --------------------------------------------------------------------------------------------
# T3 -- the UVtail subclass no longer overrides doubleIntegrationSymmetric. This locks in that
# the inherited path reproduces the deleted override's closed form bulk + 2 A T + T^2 for a
# multi-row constraint with a row-independent tail moment.
# --------------------------------------------------------------------------------------------

def test_symmetric_block_matches_legacy_closed_form():
    kernel = make_kernel()
    tail_value = analytic_sum_rule_tail(W_UV, GAMMA_GLUON)
    integrator = integrators.GaussLegendre_1D_log_UVtail(
        W_MIN, W_UV, INT_N, propagator_uv_shape,
        tail_moment=sum_rule_tail_moment(W_UV, GAMMA_GLUON),
    )
    p = np.array([0.7, 3.0, 11.0])  # Three rows, so a broadcast-vs-outer-product bug would show.
    constraint = constraints.LinearEquality(
        operators.Integral(kl_p2, integrator),
        {"x": p, "y": np.zeros_like(p), "cov_y": DATA_COV * np.ones_like(p)},
    )

    symmetric = integrator.doubleIntegrationSymmetric(constraint, kernel)
    general = integrator.doubleIntegration(constraint, kernel, constraint)
    np.testing.assert_allclose(symmetric, general, rtol=1e-12)

    bulk = bulk_double_integration(integrator, constraint, kernel, constraint)
    c_row = constraint(make_row_vector(integrator.w), x=make_column_vector(constraint.x))
    A = (integrator.weights * c_row @ kernel(integrator.w, integrator.w_uv)) / integrator.f_uv_anchor
    ones_col = np.ones((A.shape[0], 1))
    legacy = bulk + tail_value * (A @ ones_col.T + ones_col @ A.T) + tail_value ** 2

    np.testing.assert_allclose(symmetric, legacy, rtol=1e-12)


# --------------------------------------------------------------------------------------------
# T4 -- the analytic value itself, for both anomalous dimensions, plus a quadrature cross-check
# that extending the numerical range towards infinity really does converge onto
# bulk(w_uv) + T_sr. Convergence in log(w) is slow, hence the loose few-percent tolerance.
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("gamma, uv_shape, max_truncation_error", [
    # max_truncation_error: how far the integral truncated at 10^20 still is from bulk + T. The
    # tail decays only like (2 log w)^-gamma, so the smaller gamma the slower the convergence --
    # ~1% for the gluon, ~18% for the ghost. That slowness is exactly why the analytic tail is
    # mandatory and extending the quadrature range is not a viable alternative.
    (GAMMA_GLUON, propagator_uv_shape, 0.05),
    (GAMMA_GHOST, ghost_uv_shape, 0.20),
])
def test_tail_moment_matches_analytic_sum_rule_value(gamma, uv_shape, max_truncation_error):
    integrator = integrators.GaussLegendre_1D_log_UVtail(
        W_MIN, W_UV, INT_N, uv_shape, tail_moment=sum_rule_tail_moment(W_UV, gamma)
    )
    constraint = constraints.LinearEquality(
        operators.Integral(sr_kernel, integrator),
        {"x": np.array([0.0]), "y": np.array([0.0]), "cov_y": np.array([SR_COV])},
    )
    tail = integrator.uv_tail_moment(constraint)
    assert tail.shape == (1, 1)
    np.testing.assert_allclose(tail.item(), analytic_sum_rule_tail(W_UV, gamma), rtol=1e-14)

    # Cross-check: sum_rule integral over (W_MIN, 10^k) -> integral over (W_MIN, w_uv) + T.
    # n_nodes=2000: the two truncations use different grids, so their shared bulk parts only
    # cancel to quadrature accuracy; 2000 nodes brings that below 4e-6 relative (800 leaves 2e-2).
    def numeric_sum_rule_integral(w_max, n_nodes=2000):
        integ = integrators.GaussLegendre_1D_log(W_MIN, w_max, n_nodes)
        return float((integ.weights @ (integ.w * uv_shape(integ.w))).item())

    target = numeric_sum_rule_integral(W_UV) + analytic_sum_rule_tail(W_UV, gamma)
    shortfalls = [target - numeric_sum_rule_integral(10.0 ** k) for k in (6, 8, 12, 20)]

    # Whatever the truncated quadrature misses must be precisely the closed form's own remainder
    # above the truncation point -- the direct check of the closed form against quadrature of the
    # actual (regularised) f_uv, independent of how slowly the tail converges.
    for k, shortfall in zip((6, 8, 12, 20), shortfalls):
        np.testing.assert_allclose(shortfall, analytic_sum_rule_tail(10.0 ** k, gamma), rtol=1e-5)

    errors = [abs(s) / abs(target) for s in shortfalls]
    assert errors == sorted(errors, reverse=True), \
        f"truncated integral should approach bulk + T monotonically, got {errors}"
    assert errors[-1] < max_truncation_error, \
        f"k=20 still {errors[-1]:.3%} away from bulk + T"


# --------------------------------------------------------------------------------------------
# T5 -- end-to-end self-consistency of the posterior. predict_data() and predict() must describe
# the same model: reintegrating the posterior mean over fredipy's own nodes and weights has to
# reproduce predict_data(). This is the identity the bug broke by ~1e0.
# --------------------------------------------------------------------------------------------

def test_predict_data_matches_reintegrated_posterior_mean():
    model, c_data, c_sr, data_integrator, sr_integrator = build_model()
    n_data = c_data.x.shape[0]

    nodes = data_integrator.w
    rho, _ = model.predict(nodes)                                  # (N, 1)
    mu, _ = model.predict_data()                                   # (M + 1, 1)

    data_weighted = data_integrator.weights * kl_p2(
        make_column_vector(c_data.x), make_row_vector(nodes)
    )
    np.testing.assert_allclose(data_weighted @ rho, mu[:n_data], rtol=1e-9)

    # Sum-rule row: the reintegration only covers the bulk, so the analytic tail T * A_uv has to
    # be added back, with A_uv = rho(w_uv) / f_uv(w_uv) the UV amplitude of the posterior mean.
    sr_weighted = sr_integrator.weights * sr_kernel(
        make_column_vector(c_sr.x), make_row_vector(nodes)
    )
    rho_at_w_uv, _ = model.predict(sr_integrator.w_uv)
    tail_value = sr_integrator.uv_tail_moment(c_sr).item()
    reintegrated_sr = (sr_weighted @ rho).item()
    reintegrated_sr += tail_value * rho_at_w_uv.item() / sr_integrator.f_uv_anchor

    np.testing.assert_allclose(reintegrated_sr, mu[n_data:].item(), rtol=1e-4)


# --------------------------------------------------------------------------------------------
# T6 -- reference values captured with the pre-fix code, before the tail hook existed. A model
# whose constraints all carry plain GaussLegendre_1D_log integrators must stay bit-identical:
# this is what protects three-gluon-vert, ghost-gluon-vert and fgvert-sd from the change.
# --------------------------------------------------------------------------------------------

REFERENCE_LOG_LIKELIHOOD = -278560.62089715304
REFERENCE_PREDICT_DATA = np.array([
    0.11838105684728362, 0.2086624375442625, 0.35474911867640913, 0.5465432399942074,
    0.75084480180521496, 0.92431427465635352, 1.034895190037787, 1.0745688577007968,
    1.0557558063883334, 0.99917723311227746, 0.92389010410988703, 0.8432306828617584,
    0.76490387672674842, 0.69258282921509817, 0.62751788803143427, 0.0017991249333135784,
])
REFERENCE_PREDICT_MEAN = np.array([
    32.848601571109612, -11.554450084397104, -0.77481442881980911,
    -0.00017162581324248194, -1.0987647334001352e-09,
])
REFERENCE_PREDICT_GRID = np.array([0.01, 0.5, 2.0, 50.0, 1e4])  # Prediction points of the reference.
REFERENCE_OPKEROP_TRACE = 10.170139984656482


def test_plain_integrators_are_unchanged_by_the_tail_hook():
    model, _, _, _, _ = build_model(with_tail=False)

    np.testing.assert_allclose(model.log_likelihood(), REFERENCE_LOG_LIKELIHOOD, rtol=1e-15)
    mu, _ = model.predict_data()
    np.testing.assert_allclose(mu.ravel(), REFERENCE_PREDICT_DATA, rtol=1e-15)
    mean, _ = model.predict(REFERENCE_PREDICT_GRID)
    np.testing.assert_allclose(mean.ravel(), REFERENCE_PREDICT_MEAN, rtol=1e-15)
    trace = np.trace(model._posterior_cache["OpKerOp"])
    np.testing.assert_allclose(trace, REFERENCE_OPKEROP_TRACE, rtol=1e-15)


# --------------------------------------------------------------------------------------------
# T7 -- (**) mixes both constraints on a shared quadrature grid, so it is undefined when the two
# integrators discretise omega differently. Fail loudly rather than silently mixing grids.
# --------------------------------------------------------------------------------------------

def test_mismatched_integrator_grids_raise():
    kernel = make_kernel()
    p, G = synthetic_data()

    data_integrator = integrators.GaussLegendre_1D_log(W_MIN, 1e4, INT_N + 30)  # Different grid.
    c_data = constraints.LinearEquality(
        operators.Integral(kl_p2, data_integrator),
        {"x": p, "y": G, "cov_y": DATA_COV * np.ones_like(G)},
    )
    sr_integrator = integrators.GaussLegendre_1D_log_UVtail(
        W_MIN, W_UV, INT_N, propagator_uv_shape,
        tail_moment=sum_rule_tail_moment(W_UV, GAMMA_GLUON),
    )
    c_sr = constraints.LinearEquality(
        operators.Integral(sr_kernel, sr_integrator),
        {"x": np.array([0.0]), "y": np.array([0.0]), "cov_y": np.array([SR_COV])},
    )

    with pytest.raises(NotImplementedError) as excinfo:
        TwoSided()(kernel, [c_data, c_sr])
    message = str(excinfo.value)
    assert "10000.0" in message and "100000.0" in message, message
    assert str(INT_N) in message and str(INT_N + 30) in message, message


# --------------------------------------------------------------------------------------------
# T8 -- singleIntegration's tail is T[m] * K(w_uv, w_pred) / f_uv(w_uv), an outer product. The
# pre-fix code broadcast a single scalar, which happened to be right only because the sum-rule
# constraint has exactly one row.
# --------------------------------------------------------------------------------------------

def test_single_integration_tail_is_row_wise_outer_product():
    kernel = make_kernel()
    row_moments = np.array([[0.25], [0.5], [1.75]])  # Deliberately row-dependent tail moments.
    integrator = integrators.GaussLegendre_1D_log_UVtail(
        W_MIN, W_UV, INT_N, propagator_uv_shape,
        tail_moment=lambda x: row_moments.copy(),
    )
    p = np.array([0.7, 3.0, 11.0])
    constraint = constraints.LinearEquality(
        operators.Integral(kl_p2, integrator),
        {"x": p, "y": np.zeros_like(p), "cov_y": DATA_COV * np.ones_like(p)},
    )
    w_pred = make_column_vector(np.geomspace(0.05, 500.0, 7))

    full = integrator.singleIntegration(constraint, kernel, w_pred)
    assert full.shape == (3, w_pred.shape[0])

    bulk = (
        integrator.weights * constraint(make_row_vector(integrator.w), x=make_column_vector(p))
        @ kernel(integrator.w, w_pred)
    )
    anchor_row = kernel(integrator.w_uv, w_pred) / integrator.f_uv_anchor  # (1, N_pred)
    for m in range(3):
        expected = bulk[m] + row_moments[m, 0] * anchor_row.ravel()
        np.testing.assert_allclose(full[m], expected, rtol=1e-12)


# --------------------------------------------------------------------------------------------
# T9 -- the analytic NLL gradient against finite differences of the NLL itself.
#
# log_likelihood_grad() reassembles OpKerOp with dK/dtheta substituted for K. The A terms of
# (**) are linear in the kernel and differentiate themselves correctly, but T_1 T_2^T contains
# no kernel at all -- T is the analytic tail moment, a function of w_uv and gamma only -- so its
# hyperparameter derivative is zero. Adding it unchanged (which is what the code did before the
# derivative flag existed) put the gradient off by factors of 34 to 1675 on this very model, with
# the wrong sign on the RBF lengthscale. This test is the acceptance criterion for that fix: no
# other test in either suite can see a wrong gradient, since the likelihood *value* is correct
# either way.
# --------------------------------------------------------------------------------------------

FD_REL_STEP = 1e-4    # Relative central-difference step; near the optimum of truncation vs roundoff.
FD_RTOL = 1e-5        # Agreement demanded per component; the FD truncation floor here is ~5e-7.

# Hyperparameter vector [rbf_variance, rbf_lengthscale, mu_uv, l_uv] at which the check is run.
FD_PARAMS = [RBF_VARIANCE, RBF_LENGTHSCALE, MU_UV, L_UV]


def _set_flat_params(model, params):
    """Push a flat [rbf_variance, rbf_lengthscale, mu_uv, l_uv] vector into the model."""
    model.reset()
    model.kernel.set_params(kernel_params=list(params[:2]), asymp_params=list(params[2:]))


def _finite_difference_gradient(model, params, rel_step=FD_REL_STEP):
    """Central-difference gradient of log_likelihood() w.r.t. every entry of ``params``."""
    grad = []
    for i in range(len(params)):
        h = rel_step * abs(params[i])
        plus = list(params)
        plus[i] += h
        _set_flat_params(model, plus)
        f_plus = model.log_likelihood()
        minus = list(params)
        minus[i] -= h
        _set_flat_params(model, minus)
        f_minus = model.log_likelihood()
        grad.append((f_plus - f_minus) / (2.0 * h))
    _set_flat_params(model, params)
    return np.array(grad)


@pytest.mark.parametrize("with_tail", [True, False])
def test_log_likelihood_grad_matches_finite_differences(with_tail):
    """Every active hyperparameter, with and without the UVtail sum-rule constraint."""
    model, _, _, _, _ = build_model(with_tail=with_tail)
    assert model.kernel.dim == len(FD_PARAMS)

    _set_flat_params(model, FD_PARAMS)
    analytic = np.array(model.log_likelihood_grad(), dtype=float)
    numeric = _finite_difference_gradient(model, FD_PARAMS)

    np.testing.assert_allclose(analytic, numeric, rtol=FD_RTOL)


def test_gradient_flag_is_a_noop_for_plain_integrators():
    """The derivative flag must not perturb the plain-quadrature gradient path at all.

    Plain ``GaussLegendre_1D_log`` blocks are linear in the kernel, so the assembled gradient
    covariance has to be bit-identical with and without the flag. This is what guarantees that
    three-gluon-vert, ghost-gluon-vert and fgvert-sd optimize against exactly the gradients they
    did before the flag was introduced.
    """
    model, _, _, _, _ = build_model(with_tail=False)
    _set_flat_params(model, FD_PARAMS)
    for kernel_grad in model.kernel.params_gradient():
        with_flag = TwoSided()(kernel_grad, model.constraints, derivative=True)
        without_flag = TwoSided()(kernel_grad, model.constraints, derivative=False)
        assert np.array_equal(with_flag, without_flag)


def test_gradient_flag_drops_only_the_constant_tail_term():
    """With a UVtail constraint the flag must remove exactly T_1 T_2^T and nothing else."""
    model, c_data, c_sr, _, sr_integrator = build_model(with_tail=True)
    _set_flat_params(model, FD_PARAMS)
    n_data = c_data.x.shape[0]
    tail = sr_integrator.uv_tail_moment(c_sr)                     # (1, 1)

    for kernel_grad in model.kernel.params_gradient():
        with_flag = TwoSided()(kernel_grad, model.constraints, derivative=True)
        without_flag = TwoSided()(kernel_grad, model.constraints, derivative=False)
        difference = without_flag - with_flag

        # Only the sum-rule diagonal block differs, by exactly T^2: the data constraint carries a
        # plain integrator, hence a zero tail moment, hence no constant term in any block it enters.
        expected = np.zeros_like(difference)
        expected[n_data:, n_data:] = tail @ tail.T
        # rtol 1e-12: the sum-rule block is a difference of two O(1) sums, so it carries a few
        # ulp of cancellation. atol 0: every other block must be bitwise unchanged by the flag.
        np.testing.assert_allclose(difference, expected, rtol=1e-12, atol=0.0)
