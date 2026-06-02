from __future__ import annotations
from typing import TYPE_CHECKING, Callable
if TYPE_CHECKING:
    from .constraints import LinearEquality

import numpy as np

from .util import make_column_vector, make_row_vector


class Integrator():

    def __init__(
            self,
            w_min: float = 0.,
            w_max: float = 10.,
            int_n: int = 1000
            ) -> None:
        raise NotImplementedError('Need to define __init__.')

    def doubleIntegrationSymmetric(
            self,
            constraint: LinearEquality,
            kernel: Callable
            ) -> np.ndarray:
        r"""Symmetric double integration of a constraint C with a GP kernel K, i.e.

        \int_{w_{min}}^{w_{max}} \int_{w_{min}}^{w_{max}} C(w, p) K(w, w') C(w', p) dw dw'

        Parameters
        ----------
        constraint  : Some integral constraint
        kernel      : Some Gaussian process kernel

        Returns
        -------
        2D array of shape (len(constraint.x), len(constraint.x))
        """
        raise NotImplementedError

    def doubleIntegration(
            self,
            constraint1: LinearEquality,
            kernel: Callable,
            constraint2: LinearEquality
            ) -> np.ndarray:
        r"""Double integration of two (different) constraints C1 and C2 with a GP kernel K, i.e.

        \int_{w_{min}}^{w_{max}} C_1(w, p1) K(w, w') C_2(w', p2) dw'

        Parameters
        ----------
        constraint1, constraint2  : Some integral constraints
        kernel      : Some Gaussian process kernel

        Returns
        -------
        2D array of shape (len(constraint.x1), len(constraint.x2)), where
        """
        raise NotImplementedError

    def singleIntegration(
            self,
            constraint: LinearEquality,
            kernel: Callable,
            w_pred: np.ndarray
            ) -> np.ndarray:
        r"""Single integration of a constraint C with a GP kernel K, i.e.

        \int_{w_{min}}^{w_{max}} C(w, p) K(w, w_pred) dw

        Parameters
        ----------
        constraint  : Some integral constraint
        kernel      : Some Gaussian process kernel
        w_pred      : Prediction points

        Returns
        -------
        2D array of shape (len(constraint.x), len(w_pred))
        """
        raise NotImplementedError


class Riemann(Integrator):
    """Implementation of the Riemann integration in arbitrary dimensions and kernels.

    Only use this when doing higher dimensional reconstruction.
    Otherwise, use the Riemann_1D class.
    """

    def __init__(
            self,
            w_min: float = 0.,
            w_max: float = 10.,
            int_n: int = 1000
            ) -> None:

        self.w = make_column_vector(np.linspace(w_min, w_max, int_n+1))
        self.dw = (self.w[1] - self.w[0]).item()
        self.w = self.w[:-1] + 0.5 * self.dw  # midpoint rule

    def doubleIntegrationSymmetric(
            self,
            constraint: LinearEquality,
            kernel: Callable
            ) -> np.ndarray:

        p = constraint.x
        len_p = p.shape[0]
        tmp = np.zeros((len_p, len_p))
        ones_w = np.ones_like(self.w)
        for i in range(len_p):
            for j in range(i + 1):
                tmp[i, j] = self.dw**2 * (
                    constraint(make_row_vector(self.w), x=make_column_vector(np.array([p[i, 0]])))
                    @ kernel(np.c_[self.w, p[i, 1:] * ones_w], np.c_[self.w, p[j, 1:] * ones_w])
                    @ constraint(make_column_vector(self.w), x=make_row_vector(np.array([p[j, 0]])))
                ).item()

        return tmp + tmp.T - np.diag(tmp.diagonal())

    def singleIntegration(
            self,
            constraint: LinearEquality,
            kernel: Callable,
            w_pred: np.ndarray
            ) -> np.ndarray:

        p = constraint.x
        len_p = p.shape[0]
        tmp = np.zeros((len_p, w_pred.shape[0]))
        ones_w = np.ones_like(self.w)

        for i in range(len_p):
            tmp[[i], :] = self.dw * (
                constraint(make_row_vector(self.w), x=make_column_vector(np.array([p[i, 0]])))
                @ kernel(np.c_[self.w, p[i, 1:] * ones_w], w_pred)
            )

        return tmp


class Riemann_1D(Integrator):
    """Riemann Integration in exactly 1 Dimension

    Significantly faster than the more general Riemann implementation.
    """

    def __init__(
            self,
            w_min: float = 0.,
            w_max: float = 10.,
            int_n: int = 1000
            ) -> None:

        self.w = make_column_vector(np.linspace(w_min, w_max, int_n + 1))
        self.dw = self.w[1] - self.w[0]
        self.w = self.w[:-1] + 0.5 * self.dw  # midpoint rule

    def doubleIntegrationSymmetric(
            self,
            constraint: LinearEquality,
            kernel: Callable
            ) -> np.ndarray:
        return self.doubleIntegration(constraint, kernel, constraint)

    def doubleIntegration(
            self,
            constraint1: LinearEquality,
            kernel: Callable,
            constraint2: LinearEquality
            ) -> np.ndarray:
        return self.dw**2 * (
            constraint1(make_row_vector(self.w), x=make_column_vector(constraint1.x))
            @ kernel(self.w, self.w)
            @ (constraint2(make_row_vector(self.w), x=make_column_vector(constraint2.x))).T
        )

    def singleIntegration(
            self,
            constraint: LinearEquality,
            kernel: Callable,
            w_pred: np.ndarray
            ) -> np.ndarray:

        return self.dw * (
            constraint(make_row_vector(self.w), x=make_column_vector(constraint.x))
            @ kernel(self.w, w_pred)
        )


class Riemann_1D_log(Integrator):
    """Riemann integration in 1D on a logarithmic (geomspace) grid.

    Uses midpoints equally spaced in log-space and a scalar step
    ``dw = d(log ω) = log(w_max/w_min) / int_n``.

    The Jacobian of the substitution t = log ω is ω and is applied
    internally, so plain kernels ``K(p, ω)`` can be passed directly.
    """

    def __init__(
            self,
            w_min: float = 0.01,
            w_max: float = 10.,
            int_n: int = 1000
            ) -> None:

        w_edges = make_column_vector(np.geomspace(w_min, w_max, int_n + 1))
        self.dw = np.log(w_max / w_min) / int_n  # uniform step in log-space
        self.w = np.sqrt(w_edges[:-1] * w_edges[1:])  # geometric midpoints
        self.jac = make_row_vector(self.w)  # log-space Jacobian: ω_i, shape (1, n)

    def doubleIntegrationSymmetric(
            self,
            constraint: LinearEquality,
            kernel: Callable
            ) -> np.ndarray:
        return self.doubleIntegration(constraint, kernel, constraint)

    def doubleIntegration(
            self,
            constraint1: LinearEquality,
            kernel: Callable,
            constraint2: LinearEquality
            ) -> np.ndarray:
        return self.dw**2 * (
            self.jac * constraint1(make_row_vector(self.w), x=make_column_vector(constraint1.x))
            @ kernel(self.w, self.w)
            @ (self.jac * constraint2(make_row_vector(self.w), x=make_column_vector(constraint2.x))).T
        )

    def singleIntegration(
            self,
            constraint: LinearEquality,
            kernel: Callable,
            w_pred: np.ndarray
            ) -> np.ndarray:
        return self.dw * (
            self.jac * constraint(make_row_vector(self.w), x=make_column_vector(constraint.x))
            @ kernel(self.w, w_pred)
        )


class GaussLegendre_1D_log(Integrator):
    """Gauss-Legendre quadrature in 1D on a logarithmic grid.

    Maps the standard Gauss-Legendre nodes from [-1, 1] to log-space
    [log(w_min), log(w_max)] and then to ω-space via exponentiation.

    Achieves exponential convergence in the number of quadrature points for
    smooth integrands, requiring far fewer points than Riemann rules for the
    same accuracy (typically 50–200 instead of 1000).

    The log-space Jacobian factor ω is absorbed into the quadrature weights,
    so plain kernels ``K(p, ω)`` can be passed directly.
    """

    def __init__(
            self,
            w_min: float = 0.01,
            w_max: float = 10.,
            int_n: int = 100
            ) -> None:

        # GL nodes (xi in [-1,1]) and weights
        xi, wi = np.polynomial.legendre.leggauss(int_n)

        # Map nodes from [-1,1] to log-space [log(w_min), log(w_max)]
        log_min = np.log(w_min)
        log_max = np.log(w_max)
        half_range = 0.5 * (log_max - log_min)

        t = half_range * xi + 0.5 * (log_max + log_min)   # nodes in log-space
        self.w = make_column_vector(np.exp(t))              # nodes in ω-space
        # Weights absorb the half-range Jacobian and the log-space Jacobian ω
        self.weights = make_row_vector(wi * half_range) * self.w.T  # shape (1, int_n)

    def doubleIntegrationSymmetric(
            self,
            constraint: LinearEquality,
            kernel: Callable
            ) -> np.ndarray:
        return self.doubleIntegration(constraint, kernel, constraint)

    def doubleIntegration(
            self,
            constraint1: LinearEquality,
            kernel: Callable,
            constraint2: LinearEquality
            ) -> np.ndarray:
        return (
            self.weights * constraint1(make_row_vector(self.w), x=make_column_vector(constraint1.x))
            @ kernel(self.w, self.w)
            @ (self.weights * constraint2(make_row_vector(self.w), x=make_column_vector(constraint2.x))).T
        )

    def singleIntegration(
            self,
            constraint: LinearEquality,
            kernel: Callable,
            w_pred: np.ndarray
            ) -> np.ndarray:
        return (
            self.weights * constraint(make_row_vector(self.w), x=make_column_vector(constraint.x))
            @ kernel(self.w, w_pred)
        )


class GaussLegendre_1D_log_UVtail(GaussLegendre_1D_log):
    """GL log-space quadrature on (w_min, w_uv) with an analytic UV tail.

    For ω > w_uv the AsymptoticKernel factorises as
        K(ω, ω') ≈ f_uv(ω) f_uv(ω'),
    so the analytic tail T = ∫_{log w_uv}^∞ (2t)^{-35/22} dt can be
    folded in exactly via corrections to the integration methods.

    The substitution u = (2 log ω)^{-13/22} maps the UV integrand to a
    constant, giving the closed form

        T = (2 log w_uv)^{-13/22} / (2 × 13/22).

    As with the parent class, plain kernels ``K(p, ω)`` are passed directly;
    the log-space Jacobian ω is absorbed into the quadrature weights.

    **Correction formulas** (A_θ = UV-component of the bulk integral):

    * ``doubleIntegrationSymmetric``:
        full = bulk + 2 A_θ T + T²
        where A_θ[m] ≈ (1/f_uv(w_uv)) × Σ_i W_i C_m(ω_i) K(ω_i, w_uv)

    * ``singleIntegration``:
        full = numerical + T × K(w_uv, ω_pred) / f_uv(w_uv)
        (K(w_uv, ω_pred)/f_uv(w_uv) ≈ θ_uv(ω_pred) f_uv(ω_pred) for
        prediction points far from w_uv, where the RBF part of K vanishes)

    Parameters
    ----------
    w_min   : float
    w_uv    : float
        UV split where UV asymptotics fully apply.  Recommended: ≥ 1000.
    int_n   : int
    uv_func : callable
        The UV asymptotic function f_uv(ω), e.g. ``uv_asymptotics``.
    """

    def __init__(
            self,
            w_min: float,
            w_uv: float,
            int_n: int,
            uv_func: Callable
            ) -> None:
        super().__init__(w_min, w_uv, int_n)
        self.uv_func = uv_func
        # Anchor point at the UV split boundary
        self.w_uv = make_column_vector(np.array([w_uv]))      # (1, 1)
        self.f_uv_anchor = float(uv_func(self.w_uv).flat[0])   # scalar: f_uv(w_uv)
        # Analytic tail: ∫_{log w_uv}^∞ (2t)^{-35/22} dt
        self.T_uv = (2.0 * np.log(w_uv)) ** (-13.0 / 22.0) / (2.0 * 13.0 / 22.0)

    def doubleIntegrationSymmetric(
            self,
            constraint: LinearEquality,
            kernel: Callable
            ) -> np.ndarray:
        bulk = super().doubleIntegrationSymmetric(constraint, kernel)  # (M, M)

        # A_θ ≈ (W C @ K_col) / f_uv(w_uv),  shape (M, 1)
        # K_col[i] = K(ω_i, w_uv) ≈ θ_uv(ω_i) f_uv(ω_i) f_uv(w_uv)  (for large w_uv)
        K_col = kernel(self.w, self.w_uv)                              # (N, 1)
        c_row = constraint(make_row_vector(self.w), x=make_column_vector(constraint.x))
        A = (self.weights * c_row @ K_col) / self.f_uv_anchor         # (M, 1)

        # correction_{mn} = T (A_m + A_n) + T²
        ones_col = np.ones((A.shape[0], 1))
        correction = self.T_uv * (A @ ones_col.T + ones_col @ A.T) + self.T_uv ** 2
        return bulk + correction

    def singleIntegration(
            self,
            constraint: LinearEquality,
            kernel: Callable,
            w_pred: np.ndarray
            ) -> np.ndarray:
        numerical = super().singleIntegration(constraint, kernel, w_pred)
        # tail = T × K(w_uv, ω_pred) / f_uv(w_uv),  shape (1, N_pred)
        # K(w_uv, ω_pred) / f_uv(w_uv) ≈ θ_uv(ω_pred) f_uv(ω_pred) for w_uv >> ω_pred
        tail = (self.T_uv / self.f_uv_anchor) * kernel(self.w_uv, w_pred)  # (1, N_pred)
        return numerical + tail


class GaussLegendre_1D_semiinf(Integrator):
    """Gauss-Legendre quadrature on the full semi-infinite interval (0, ∞).

    Uses the rational change of variables

        ω = w_scale · (1 + u) / (1 − u),   u ∈ (−1, 1)

    which maps (−1, 1) exactly onto (0, ∞).  The log-space Jacobian
    d(log ω)/du = 2/(1 − u²) and the ω factor are both absorbed into
    the quadrature weights, so plain kernels ``K(p, ω)`` can be passed
    directly.

    Parameters
    ----------
    w_scale : float
        The scale point: u = 0 maps to ω = w_scale.  Choose it near the
        centre of the integrand in log-space for the best node distribution.
    int_n   : int
        Number of Gauss-Legendre nodes.

    Notes
    -----
    For n = 100 nodes and w_scale = 1, the outermost nodes lie at
    roughly ω ≈ 7×10⁻⁵ and ω ≈ 1.4×10⁴.  The GL weights at those
    extreme nodes automatically vanish proportionally to (1 − u²),
    exactly cancelling the Jacobian divergence, so the product weight
    W_i = w_i · 2/(1 − u_i²) is bounded O(1/n²) at all nodes.
    """

    def __init__(
            self,
            w_scale: float = 1.0,
            int_n: int = 100
            ) -> None:

        xi, wi = np.polynomial.legendre.leggauss(int_n)

        # rational map (-1,1) → (0,∞)
        self.w = make_column_vector(w_scale * (1.0 + xi) / (1.0 - xi))
        # weights: log-space d(log ω)/du = 2/(1-u²), plus ω Jacobian absorbed
        self.weights = make_row_vector(wi * 2.0 / (1.0 - xi**2)) * self.w.T

    def doubleIntegrationSymmetric(
            self,
            constraint: LinearEquality,
            kernel: Callable
            ) -> np.ndarray:
        return self.doubleIntegration(constraint, kernel, constraint)

    def doubleIntegration(
            self,
            constraint1: LinearEquality,
            kernel: Callable,
            constraint2: LinearEquality
            ) -> np.ndarray:
        return (
            self.weights * constraint1(make_row_vector(self.w), x=make_column_vector(constraint1.x))
            @ kernel(self.w, self.w)
            @ (self.weights * constraint2(make_row_vector(self.w), x=make_column_vector(constraint2.x))).T
        )

    def singleIntegration(
            self,
            constraint: LinearEquality,
            kernel: Callable,
            w_pred: np.ndarray
            ) -> np.ndarray:
        return (
            self.weights * constraint(make_row_vector(self.w), x=make_column_vector(constraint.x))
            @ kernel(self.w, w_pred)
        )


class Simpson_1D(Integrator):
    """Implementation of Simpson's rule for 1D integration"""

    def __init__(
            self,
            w_min: float = 0.,
            w_max: float = 10.,
            int_n: int = 1000
            ) -> None:

        if int_n % 2 == 0:
            int_n = int_n + 1
        self.w = make_column_vector(np.linspace(w_min, w_max, int_n))
        self.dw = self.w[1] - self.w[0]
        # construct array of alternating prefactors
        self.prefactors = np.empty_like(self.w)
        self.prefactors[::2] = 2
        self.prefactors[1::2] = 4
        self.prefactors[0] = 1
        self.prefactors[-1] = 1
        self.prefactors = make_row_vector(self.prefactors)

    def doubleIntegrationSymmetric(
            self,
            constraint: LinearEquality,
            kernel: Callable
            ) -> np.ndarray:
        return self.doubleIntegration(constraint, kernel, constraint)

    def doubleIntegration(
            self,
            constraint1: LinearEquality,
            kernel: Callable,
            constraint2: LinearEquality
            ) -> np.ndarray:

        return self.dw**2 / 9 * (
            self.prefactors * constraint1(make_row_vector(self.w), x=make_column_vector(constraint1.x))
            @ kernel(self.w, self.w)
            @ (self.prefactors.T * constraint2(make_column_vector(self.w), x=make_row_vector(constraint2.x)))
        )

    def singleIntegration(
            self,
            constraint: LinearEquality,
            kernel: Callable,
            w_pred: np.ndarray
            ) -> np.ndarray:

        return self.dw / 3 * (
            self.prefactors * constraint(make_row_vector(self.w), x=make_column_vector(constraint.x))
            @ kernel(self.w, w_pred)
        )


class Simpson(Integrator):
    """Implementation of Simpson's rule for higher dimensional integration"""

    def __init__(
            self,
            w_min: float = 0.,
            w_max: float = 10.,
            int_n: int = 1000
            ) -> None:

        if int_n % 2 == 0:
            int_n = int_n + 1
        self.w = make_column_vector(np.linspace(w_min, w_max, int_n))
        self.dw = (self.w[1] - self.w[0]).item()
        # construct array of alternating prefactors
        self.prefactors = np.empty_like(self.w)
        self.prefactors[::2] = 2
        self.prefactors[1::2] = 4
        self.prefactors[0] = 1
        self.prefactors[-1] = 1
        self.prefactors = make_row_vector(self.prefactors)

    def doubleIntegrationSymmetric(
            self,
            constraint: LinearEquality,
            kernel: Callable
            ) -> np.ndarray:

        p = constraint.x
        len_p = p.shape[0]
        tmp = np.zeros((len_p, len_p))
        ones_w = np.ones_like(self.w)

        for i in range(len_p):
            for j in range(i + 1):
                tmp[i, j] = self.dw**2 / 9 * (
                    self.prefactors
                    * constraint(make_row_vector(self.w), x=make_column_vector(np.array([p[i, 0]])))
                    @ kernel(np.c_[self.w, p[i, 1:] * ones_w], np.c_[self.w, p[j, 1:] * ones_w])
                    @ (self.prefactors
                       * constraint(make_row_vector(self.w), x=make_column_vector(np.array([p[j, 0]])))).T
                ).item()

        return tmp + tmp.T - np.diag(tmp.diagonal())

    def singleIntegration(
            self,
            constraint: LinearEquality,
            kernel: Callable,
            w_pred: np.ndarray
            ) -> np.ndarray:

        p = constraint.x
        len_p = p.shape[0]
        tmp = np.zeros((len_p, w_pred.shape[0]))
        ones_w = np.ones_like(self.w)

        for i in range(len_p):
            tmp[[i], :] = self.dw / 3 * (
                self.prefactors * constraint(make_row_vector(self.w), x=make_column_vector(np.array([p[i, 0]])))
                @ kernel(np.c_[self.w, p[i, 1:] * ones_w], w_pred)
            )

        return tmp
