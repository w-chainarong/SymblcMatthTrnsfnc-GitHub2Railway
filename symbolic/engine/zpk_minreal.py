# ==================================================
# zpk_minreal.py
# STEP 5: Numeric ZPK + Minreal Engine
# ==================================================

from sympy import (
    fraction, Poly, Symbol, expand,
    nroots, re, im
)
import math


class ZPKMinrealEngine:
    """
    STEP 5: Numeric ZPK + minreal Engine (MATLAB-like)

    Responsibilities:
    - Convert transfer function to ZPK form numerically
    - Perform numeric pole-zero cancellation (minreal)
    - Format result in MATLAB-like factorized form
    - Provide LaTeX-ready output

    IMPORTANT DESIGN RULES:
    - NUMERIC / HEURISTIC ONLY
    - NO symbolic modification
    - NO topology changes
    - Safe for UI / presentation layer
    """

    # --------------------------------------------------
    # Constructor
    # --------------------------------------------------

    def __init__(self, tf_expr, s: Symbol):
        """
        Parameters
        ----------
        tf_expr : sympy expression
            Overall transfer function (already validated upstream)
        s : sympy Symbol
            Laplace variable
        """
        self.tf = tf_expr
        self.s = s

        self.zeros = []     # list[complex]
        self.poles = []     # list[complex]
        self.gain = 1.0

        self._zpk_done = False
        self._minreal_done = False

    # --------------------------------------------------
    # STEP 1: ZPK extraction (numeric)
    # --------------------------------------------------

    def compute_zpk(self):
        """
        Extract zeros, poles, and gain numerically.

        Returns
        -------
        zeros, poles, gain
        """
        num, den = fraction(self.tf)

        num = expand(num)
        den = expand(den)

        p_num = Poly(num, self.s)
        p_den = Poly(den, self.s)

        # ---- Gain (leading coefficient ratio) ----
        try:
            self.gain = float(p_num.LC() / p_den.LC())
        except Exception:
            self.gain = 1.0

        # ---- Zeros ----
        if p_num.degree() > 0:
            self.zeros = list(nroots(p_num))
        else:
            self.zeros = []

        # ---- Poles ----
        if p_den.degree() > 0:
            self.poles = list(nroots(p_den))
        else:
            self.poles = []

        self._zpk_done = True
        return self.zeros, self.poles, self.gain

    # --------------------------------------------------
    # STEP 2: minreal (numeric cancellation)
    # --------------------------------------------------

    def minreal(self, tol: float = 1e-3):
        """
        Numeric pole-zero cancellation (MATLAB-like)

        Parameters
        ----------
        tol : float
            Cancellation tolerance

        Returns
        -------
        zeros, poles, gain
        """
        if not self._zpk_done:
            self.compute_zpk()

        remaining_zeros = []
        remaining_poles = self.poles.copy()

        for z in self.zeros:
            cancelled = False
            for p in remaining_poles:
                if abs(complex(z) - complex(p)) < tol:
                    remaining_poles.remove(p)
                    cancelled = True
                    break
            if not cancelled:
                remaining_zeros.append(z)

        self.zeros = remaining_zeros
        self.poles = remaining_poles
        self._minreal_done = True

        return self.zeros, self.poles, self.gain

    # --------------------------------------------------
    # Internal helpers: factor formatting
    # --------------------------------------------------

    def _linear_factor(self, root, precision=3, tol=1e-6):
        """
        Format (s + a) or (s - a) for (almost) real root
        """
        if abs(im(root)) > tol:
            raise ValueError("Complex root passed to linear factor")

        a = -float(re(root))

        if abs(a) < 10**(-precision):
            return "s"

        sign = "+" if a >= 0 else "-"
        return f"(s {sign} {abs(a):.{precision}g})"

    def _quadratic_factor(self, r1, r2, precision=3):
        """
        Format (s^2 + a s + b) from conjugate pair
        """
        r1 = complex(r1)
        r2 = complex(r2)

        a = -(r1 + r2).real
        b = (r1 * r2).real

        a = round(a, precision)
        b = round(b, precision)

        return f"(s^2 + {a} s + {b})"

    # --------------------------------------------------
    # STEP 3: MATLAB-like output
    # --------------------------------------------------

    def to_matlab_like(self, precision=3):
        """
        Return MATLAB-like factorized string
        """
        if not self._minreal_done:
            self.minreal()

        zeros = self.zeros.copy()
        poles = self.poles.copy()

        num_factors = []
        den_factors = []

        # ---- Zeros ----
        used = set()
        for i, z in enumerate(zeros):
            if i in used:
                continue

            if abs(im(z)) < 1e-6:
                num_factors.append(self._linear_factor(z, precision))
            else:
                for j in range(i + 1, len(zeros)):
                    if abs(z.conjugate() - zeros[j]) < 1e-6:
                        num_factors.append(
                            self._quadratic_factor(z, zeros[j], precision)
                        )
                        used.add(j)
                        break
            used.add(i)

        # ---- Poles ----
        used = set()
        for i, p in enumerate(poles):
            if i in used:
                continue

            if abs(im(p)) < 1e-6:
                den_factors.append(self._linear_factor(p, precision))
            else:
                for j in range(i + 1, len(poles)):
                    if abs(p.conjugate() - poles[j]) < 1e-6:
                        den_factors.append(
                            self._quadratic_factor(p, poles[j], precision)
                        )
                        used.add(j)
                        break
            used.add(i)

        num = "".join(num_factors) if num_factors else "1"
        den = "".join(den_factors) if den_factors else "1"
        k = f"{self.gain:.6g}"

        bar = "-" * max(len(num), len(den))
        return f"{k}{num}\n{bar}\n{den}"

    # --------------------------------------------------
    # STEP 4: LaTeX output
    # --------------------------------------------------

    def to_latex(self, precision=3):
        """
        Return LaTeX version of MATLAB-like output
        """
        matlab_like = self.to_matlab_like(precision)
        lines = matlab_like.splitlines()

        if len(lines) != 3:
            return matlab_like

        num = lines[0]
        den = lines[2]

        return r"\frac{" + num + "}{" + den + "}"
