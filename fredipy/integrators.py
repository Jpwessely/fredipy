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
            kernel: Callable,
            derivative: bool = False
            ) -> np.ndarray:
        r"""Symmetric double integration of a constraint C with a GP kernel K, i.e.

        \int_{w_{min}}^{w_{max}} \int_{w_{min}}^{w_{max}} C(w, p) K(w, w') C(w', p) dw dw'

        Parameters
        ----------
        constraint  : Some integral constraint
        kernel      : Some Gaussian process kernel
        derivative  : True when ``kernel`` is dK/dtheta rather than K, see
                      :meth:`doubleIntegration`.

        Returns
        -------
        2D array of shape (len(constraint.x), len(constraint.x))
        """
        raise NotImplementedError

    def doubleIntegration(
            self,
            constraint1: LinearEquality,
            kernel: Callable,
            constraint2: LinearEquality,
            derivative: bool = False
            ) -> np.ndarray:
        r"""Double integration of two (different) constraints C1 and C2 with a GP kernel K, i.e.

        \int_{w_{min}}^{w_{max}} C_1(w, p1) K(w, w') C_2(w', p2) dw'

        Parameters
        ----------
        constraint1, constraint2  : Some integral constraints
        kernel      : Some Gaussian process kernel
        derivative  : Flag set by ``GaussianProcess.log_likelihood_grad`` (via
                      ``TwoSided``) to say that ``kernel`` is the derivative
                      dK/dtheta of the kernel with respect to one hyperparameter
                      rather than the kernel itself.  Purely quadrature-based
                      implementations are linear in ``kernel`` and can ignore it;
                      any term that is *not* linear in ``kernel`` (the constant
                      analytic tail moment of
                      :meth:`GaussLegendre_1D_log._uv_tail_correction`) must be
                      dropped when it is set, since the derivative of a constant
                      is zero.

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

    def uv_tail_moment(
            self,
            constraint: LinearEquality
            ) -> np.ndarray:
        r"""Analytic UV tail moment ``T`` of this integrator for one constraint.

        .. math::
            T_i[m] = \int_{w_{uv}}^{\infty} C_i(p_m, w) f_{uv}(w)\, dw

        i.e. the constraint kernel weighted by the *bare* UV asymptotic shape
        ``f_uv`` over the part of the ω axis that the quadrature grid does not
        cover.  It is the only ingredient (besides the bulk-UV overlap ``A``,
        see :meth:`uv_anchor`) of the general two-constraint tail correction

        .. math::
            \Sigma_{12} = BB + A_1 T_2^T + T_1 A_2^T + T_1 T_2^T   \qquad (**)

        implemented in :meth:`GaussLegendre_1D_log._uv_tail_correction`.

        ``T`` is per-constraint **and** per-row: two constraints whose kernels
        have different UV falloffs have different tail moments, so a single
        shared scalar is *not* valid for the cross block (see
        ``docs/fredipy_uvtail_fix_plan.md`` in the reconstructions repo).

        Returns
        -------
        2D array of shape ``(len(constraint.x), 1)``.  The default is all
        zeros — integrators that carry no analytic tail, which lets
        ``_uv_tail_correction`` short-circuit and stay bit-identical to the
        uncorrected quadrature.
        """
        return np.zeros((constraint.x.shape[0], 1))

    def uv_anchor(self) -> tuple | None:
        r"""Anchor point used to evaluate the bulk-UV overlap ``A`` of ``(**)``.

        .. math::
            A_i[m] = \frac{1}{f_{uv}(w_{uv})}
                     \sum_j W_j\, C_i(p_m, w_j)\, K(w_j, w_{uv})

        which is *exact*, not approximate, whenever ``w_uv >> mu_uv``: the UV
        branch of ``AsymptoticKernel`` factorises there as
        ``K(w, w_uv) = θ_uv(w) f_uv(w) f_uv(w_uv)``, so dividing by the anchor
        value ``f_uv(w_uv)`` recovers ``Σ_j W_j C_i(w_j) θ_uv(w_j) f_uv(w_j)``
        to all digits.

        Returns
        -------
        ``None`` for tail-free integrators (the default).  Tail-carrying
        integrators return ``(w_uv, f_uv_anchor)`` with ``w_uv`` a ``(1, 1)``
        column vector holding the UV split point and ``f_uv_anchor`` the scalar
        ``f_uv(w_uv)``.
        """
        return None


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
            kernel: Callable,
            derivative: bool = False      # unused: this rule is linear in `kernel`
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
            kernel: Callable,
            derivative: bool = False      # unused: this rule is linear in `kernel`
            ) -> np.ndarray:
        return self.doubleIntegration(constraint, kernel, constraint, derivative=derivative)

    def doubleIntegration(
            self,
            constraint1: LinearEquality,
            kernel: Callable,
            constraint2: LinearEquality,
            derivative: bool = False      # unused: this rule is linear in `kernel`
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
            kernel: Callable,
            derivative: bool = False      # unused: this rule is linear in `kernel`
            ) -> np.ndarray:
        return self.doubleIntegration(constraint, kernel, constraint, derivative=derivative)

    def doubleIntegration(
            self,
            constraint1: LinearEquality,
            kernel: Callable,
            constraint2: LinearEquality,
            derivative: bool = False      # unused: this rule is linear in `kernel`
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

        # Grid definition kept as attributes purely for diagnostics: the node-compatibility
        # guard in _uv_tail_correction names these in its error message.
        self.w_min = w_min      # lower end of the quadrature range in ω
        self.w_max = w_max      # upper end of the quadrature range in ω
        self.int_n = int_n      # number of Gauss-Legendre nodes

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
            kernel: Callable,
            derivative: bool = False
            ) -> np.ndarray:
        return self.doubleIntegration(constraint, kernel, constraint, derivative=derivative)

    def doubleIntegration(
            self,
            constraint1: LinearEquality,
            kernel: Callable,
            constraint2: LinearEquality,
            derivative: bool = False
            ) -> np.ndarray:
        bulk = (
            self.weights * constraint1(make_row_vector(self.w), x=make_column_vector(constraint1.x))
            @ kernel(self.w, self.w)
            @ (self.weights * constraint2(make_row_vector(self.w), x=make_column_vector(constraint2.x))).T
        )
        # The analytic UV tail lives here (not as an override on the UVtail subclass) so that
        # both orderings of a constraint pair go through the same code and the assembled
        # covariance matrix is symmetric *by construction* — np.linalg.cholesky reads only the
        # lower triangle and would silently accept an asymmetric matrix otherwise.
        return bulk + self._uv_tail_correction(
            constraint1, kernel, constraint2, derivative=derivative)

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

    @staticmethod
    def _same_quadrature(
            integrator1: Integrator,
            integrator2: Integrator
            ) -> bool:
        """True if two integrators share (numerically) the same nodes and weights."""
        w1, w2 = getattr(integrator1, 'w', None), getattr(integrator2, 'w', None)
        v1, v2 = getattr(integrator1, 'weights', None), getattr(integrator2, 'weights', None)
        if w1 is None or w2 is None or v1 is None or v2 is None:
            return False
        if w1.shape != w2.shape or v1.shape != v2.shape:
            return False
        return bool(np.allclose(w1, w2) and np.allclose(v1, v2))

    @staticmethod
    def _describe_grid(integrator: Integrator) -> str:
        """Human-readable grid summary used in the node-mismatch error message."""
        return (f"{type(integrator).__name__}(w_min={getattr(integrator, 'w_min', '?')}, "
                f"w_max={getattr(integrator, 'w_max', '?')}, "
                f"int_n={getattr(integrator, 'int_n', '?')})")

    def _uv_tail_correction(
            self,
            constraint1: LinearEquality,
            kernel: Callable,
            constraint2: LinearEquality,
            derivative: bool = False
            ) -> np.ndarray | float:
        r"""Analytic UV tail correction to the (constraint1, constraint2) covariance block.

        Splitting *each* of the two integrals at the UV point ``w_uv`` into a bulk part
        ``(w_min, w_uv)`` — which the quadrature grid covers — and a tail part
        ``(w_uv, ∞)`` — which it does not — and using that ``AsymptoticKernel``
        factorises as ``K(w, w') = [θ_uv(w) f_uv(w)] f_uv(w')`` whenever ``w' > w_uv``
        with ``w_uv >> mu_uv`` (rank-1 in the tail, exact up to
        ``O(exp(-(w_uv - mu_uv)/l_uv))``), the four pieces give

        .. math::
            \Sigma_{12} = BB + A_1 T_2^T + T_1 A_2^T + T_1 T_2^T   \qquad (**)

        with ``BB`` the plain double quadrature computed by
        :meth:`doubleIntegration`, ``A_i`` the bulk-UV overlap (see
        :meth:`Integrator.uv_anchor`) and ``T_i`` the per-constraint, per-row tail
        moment (see :meth:`Integrator.uv_tail_moment`).

        Setting ``C_1 = C_2`` and ``T`` row-independent collapses ``(**)`` to the
        legacy symmetric closed form ``bulk + 2 A T + T²``, so the symmetric case is
        a special case of this method and needs no separate override.

        Gradient mode (``derivative=True``)
        -----------------------------------
        ``GaussianProcess.log_likelihood_grad`` reassembles the same block with
        ``dK/dtheta`` substituted for ``K``.  Under that substitution the two ``A``
        terms differentiate themselves correctly — ``A_i`` is linear in the kernel
        (``A_i = W C_i K(ω, w_uv) / f_uv(w_uv)`` with a hyperparameter-independent
        anchor value), so ``A_i[dK/dtheta] = dA_i/dtheta`` exactly.  The term
        ``T_1 T_2^T`` however contains no kernel at all: ``T`` is the *analytic*
        tail moment, a function of ``w_uv`` and the anomalous dimension only, so
        ``d(T_1 T_2^T)/dtheta = 0``.  Adding it unchanged in gradient mode would
        treat the derivative of a constant as the constant itself, which is why it
        is dropped here.  The likelihood *value* is unaffected either way; only the
        gradient is.
        """
        integrator1 = constraint1.op.integrator
        integrator2 = constraint2.op.integrator
        tail1 = integrator1.uv_tail_moment(constraint1)
        tail2 = integrator2.uv_tail_moment(constraint2)

        # Fast path: no constraint carries an analytic tail, so (**) reduces to BB.
        # This short circuit is load-bearing — it keeps every plain-quadrature model
        # (three-gluon-vert, ghost-gluon-vert, fgvert-sd, all pre-existing fredipy
        # tests) bit-identical to the uncorrected result. Do not drop it.
        if not np.any(tail1) and not np.any(tail2):
            return 0.0

        # (**) mixes the two constraints on a *shared* set of quadrature nodes; it is
        # not well defined if the two integrators discretise ω differently.
        if integrator1 is not integrator2 and not self._same_quadrature(integrator1, integrator2):
            raise NotImplementedError(
                "UV tail correction requires both integral constraints to share the same "
                "quadrature nodes and weights, but got "
                f"{self._describe_grid(integrator1)} and {self._describe_grid(integrator2)}."
            )

        anchors = [a for a in (integrator1.uv_anchor(), integrator2.uv_anchor()) if a is not None]
        if not anchors:
            raise NotImplementedError(
                "A non-zero UV tail moment was reported but no integrator supplied a UV "
                f"anchor via uv_anchor(): {self._describe_grid(integrator1)} and "
                f"{self._describe_grid(integrator2)}."
            )
        w_uv, f_uv_anchor = anchors[0]
        for other_w_uv, other_f_uv in anchors[1:]:
            assert np.allclose(w_uv, other_w_uv) and np.isclose(f_uv_anchor, other_f_uv), \
                "Integral constraints report incompatible UV anchors."

        # A_i = (W C_i) @ K(ω, w_uv) / f_uv(w_uv), both evaluated on *this* integrator's
        # nodes (guaranteed above to coincide with the other integrator's).
        c1_row = constraint1(make_row_vector(self.w), x=make_column_vector(constraint1.x))
        c2_row = constraint2(make_row_vector(self.w), x=make_column_vector(constraint2.x))
        K_col = kernel(self.w, w_uv)                                   # (N, 1)
        A1 = (self.weights * c1_row @ K_col) / f_uv_anchor             # (M1, 1)
        A2 = (self.weights * c2_row @ K_col) / f_uv_anchor             # (M2, 1)

        correction = A1 @ tail2.T + tail1 @ A2.T
        if derivative:
            # d(T_1 T_2^T)/dtheta = 0: T carries no hyperparameter dependence. See the
            # "Gradient mode" section of the docstring.
            return correction
        return correction + tail1 @ tail2.T


class GaussLegendre_1D_log_UVtail(GaussLegendre_1D_log):
    """GL log-space quadrature on (w_min, w_uv) plus an analytic UV tail.

    **Rank-1 tail assumption.**  For ω' > w_uv with w_uv >> mu_uv the soft
    thetas of ``AsymptoticKernel`` saturate (θ_uv(ω') = 1 to machine
    precision), so the kernel becomes rank-1 in the tail:

        K(ω, ω') = [θ_uv(ω) f_uv(ω)] · f_uv(ω')            (*)

    This is the *only* assumption made here; it holds up to
    ``O(exp(-(w_uv - mu_uv)/l_uv))``.

    **General two-constraint correction.**  Splitting each of the two integrals
    of a covariance block at w_uv into bulk (w_min, w_uv) and tail (w_uv, ∞) and
    applying (*) to whichever argument lies in the tail gives

        A_i[m] = ∫_bulk       C_i(p_m, ω) θ_uv(ω) f_uv(ω) dω
        T_i[m] = ∫_{w_uv}^∞   C_i(p_m, ω) f_uv(ω) dω

        Σ_12 = BB + A_1 T_2^T + T_1 A_2^T + T_1 T_2^T      (**)

    where ``BB`` is the plain double quadrature of the parent class.  Two points
    that are easy to get wrong:

    * ``T`` is **per constraint and per row**, not a single shared scalar: two
      constraints whose kernels have different UV falloffs (e.g. a sum-rule
      kernel ``C(p, ω) = ω`` versus a Källén-Lehmann data kernel) have entirely
      different tail moments.  This is why ``tail_moment`` is a required
      constructor argument rather than a hardcoded closed form.
    * ``A`` obtained via the anchor, ``A = (W C @ K(ω, w_uv)) / f_uv(w_uv)``, is
      **exact**, not approximate: by (*) the anchor column is
      ``K(ω_i, w_uv) = θ_uv(ω_i) f_uv(ω_i) f_uv(w_uv)``, so the division
      recovers the bulk-UV overlap to all digits.

    The symmetric case ``C_1 = C_2`` with row-independent ``T`` collapses (**)
    to the legacy closed form ``bulk + 2 A T + T²``, so this class deliberately
    does **not** override ``doubleIntegration``/``doubleIntegrationSymmetric``:
    the correction is applied by the parent class for *both* orderings of a
    constraint pair, which is what keeps the assembled covariance matrix
    symmetric by construction (``np.linalg.cholesky`` reads only the lower
    triangle and would silently accept an asymmetric matrix).

    As with the parent class, plain kernels ``K(p, ω)`` are passed directly;
    the log-space Jacobian ω is absorbed into the quadrature weights.

    Parameters
    ----------
    w_min   : float
        Lower end of the quadrature range in ω.
    w_uv    : float
        UV split where the UV asymptotics fully apply, i.e. the upper end of the
        quadrature range; everything above it is covered by ``tail_moment``.
        Recommended: ≥ 1000, and in any case >> mu_uv of the kernel.
    int_n   : int
        Number of Gauss-Legendre nodes on (w_min, w_uv).
    uv_func : callable
        The UV asymptotic function f_uv(ω), e.g. ``uv_asymptotics``.  Used only
        to evaluate the anchor value f_uv(w_uv).
    tail_moment : callable
        Maps ``constraint.x`` of shape (M, d) to the tail moments T of shape
        (M, 1), i.e. ``T[m] = ∫_{w_uv}^∞ C(p_m, ω) f_uv(ω) dω`` for the
        constraint this integrator is attached to.  Required (no default): a
        silent default is exactly what let a gluon-specific closed form be
        applied to the ghost sum rule, off by a factor 10.4.
    """

    def __init__(
            self,
            w_min: float,
            w_uv: float,
            int_n: int,
            uv_func: Callable,
            tail_moment: Callable[[np.ndarray], np.ndarray]
            ) -> None:
        super().__init__(w_min, w_uv, int_n)
        self.uv_func = uv_func          # UV asymptotic shape f_uv(ω), used for the anchor value
        self.tail_moment = tail_moment  # constraint.x -> (M, 1) analytic tail moments T
        # Anchor point at the UV split boundary
        self.w_uv = make_column_vector(np.array([w_uv]))      # (1, 1)
        self.f_uv_anchor = float(uv_func(self.w_uv).flat[0])   # scalar: f_uv(w_uv)

    def uv_tail_moment(
            self,
            constraint: LinearEquality
            ) -> np.ndarray:
        tail = np.asarray(self.tail_moment(constraint.x), dtype=float)
        if tail.ndim == 1:
            tail = tail.reshape(-1, 1)
        expected = (constraint.x.shape[0], 1)
        if tail.shape != expected:
            raise ValueError(
                f"tail_moment must return an array of shape {expected} for this constraint, "
                f"got {tail.shape}."
            )
        return tail

    def uv_anchor(self) -> tuple:
        return (self.w_uv, self.f_uv_anchor)

    def singleIntegration(
            self,
            constraint: LinearEquality,
            kernel: Callable,
            w_pred: np.ndarray
            ) -> np.ndarray:
        numerical = super().singleIntegration(constraint, kernel, w_pred)
        # tail[m, n] = T[m] × K(w_uv, ω_pred[n]) / f_uv(w_uv) — an outer product, since T is
        # per-row: broadcasting a single scalar T is only correct for a one-row constraint.
        # K(w_uv, ω_pred) / f_uv(w_uv) ≈ θ_uv(ω_pred) f_uv(ω_pred) for w_uv >> ω_pred.
        tail = self.uv_tail_moment(constraint) @ (kernel(self.w_uv, w_pred) / self.f_uv_anchor)
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
            kernel: Callable,
            derivative: bool = False      # unused: this rule is linear in `kernel`
            ) -> np.ndarray:
        return self.doubleIntegration(constraint, kernel, constraint, derivative=derivative)

    def doubleIntegration(
            self,
            constraint1: LinearEquality,
            kernel: Callable,
            constraint2: LinearEquality,
            derivative: bool = False      # unused: this rule is linear in `kernel`
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
            kernel: Callable,
            derivative: bool = False      # unused: this rule is linear in `kernel`
            ) -> np.ndarray:
        return self.doubleIntegration(constraint, kernel, constraint, derivative=derivative)

    def doubleIntegration(
            self,
            constraint1: LinearEquality,
            kernel: Callable,
            constraint2: LinearEquality,
            derivative: bool = False      # unused: this rule is linear in `kernel`
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
            kernel: Callable,
            derivative: bool = False      # unused: this rule is linear in `kernel`
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
