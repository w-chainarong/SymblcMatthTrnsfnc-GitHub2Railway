import numpy as np
from scipy import signal
import sympy


class NumericalEngine:
    """
    Engine for computing numerical time responses
    from a symbolic transfer function G(s).
    It also computes standard step-response metrics when applicable.
    """

    def __init__(self, s):
        self.s = s

    def _trim_leading_zeros(self, coeffs, tol=1e-14):
        """Remove leading zero coefficients while keeping at least one term."""
        trimmed = list(coeffs)
        while len(trimmed) > 1 and abs(trimmed[0]) < tol:
            trimmed.pop(0)
        return trimmed

    def _validate_numeric_expr(self, expr):
        """Check that the expression contains no unresolved symbols other than s."""
        extra_symbols = expr.free_symbols - {self.s}
        if extra_symbols:
            raise ValueError(
                "Numerical evaluation is not possible because unresolved symbols remain: "
                + ", ".join(str(x) for x in sorted(extra_symbols, key=str))
            )

    def _validate_response_arrays(self, t, y):
        """Validate numerical output arrays."""
        if t is None or y is None:
            raise ValueError("The numerical solver returned no data")

        t = np.asarray(t)
        y = np.asarray(y)

        if t.size == 0 or y.size == 0:
            raise ValueError("The numerical solver returned empty arrays")

        if not np.all(np.isfinite(t)):
            raise ValueError("Time data contains NaN or Inf")

        if not np.all(np.isfinite(y)):
            raise ValueError("Response data contains NaN or Inf")

    def _auto_time_scale(self, sys):
        """
        Estimate a suitable simulation horizon from poles.
        """
        poles = np.asarray(sys.poles, dtype=complex)

        if poles.size == 0:
            return 10.0

        stable_poles = [p for p in poles if p.real < -1e-8]
        marginal_poles = [p for p in poles if abs(p.real) <= 1e-8]
        unstable_poles = [p for p in poles if p.real > 1e-8]

        if stable_poles:
            slowest_decay = min(abs(p.real) for p in stable_poles)
            t_end = 5.0 / slowest_decay

            max_imag = max(abs(p.imag) for p in stable_poles)
            if max_imag > 1e-8:
                oscillation_window = 4.0 * (2.0 * np.pi / max_imag)
                t_end = max(t_end, oscillation_window)

            return float(np.clip(t_end, 1e-3, 1e4))

        if marginal_poles:
            return 20.0

        if unstable_poles:
            fastest_growth = max(p.real for p in unstable_poles)
            return float(np.clip(2.0 / fastest_growth, 1.0, 10.0))

        return 10.0

    def _compute_step_metrics(self, t, y, tol=0.02):
        """
        Compute standard step-response metrics:
        - final_value
        - overshoot_percent
        - rise_time (10% to 90%)
        - settling_time (within ±tol band)
        """
        t = np.asarray(t, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)

        if t.size == 0 or y.size == 0:
            return None

        y_final = y[-1]

        if not np.isfinite(y_final):
            return None

        metrics = {
            "final_value": float(y_final),
            "peak_value": None,
            "overshoot_percent": None,
            "rise_time": None,
            "settling_time": None,
            "settling_band_percent": float(tol * 100.0),
        }

        y_peak = float(np.max(y))
        metrics["peak_value"] = y_peak

        # Overshoot
        if abs(y_final) > 1e-12:
            overshoot = max(0.0, (y_peak - y_final) / abs(y_final) * 100.0)
            metrics["overshoot_percent"] = float(overshoot)
        else:
            metrics["overshoot_percent"] = None

        # Rise Time: 10% to 90% of final value
        if abs(y_final) > 1e-12:
            y10 = 0.1 * y_final
            y90 = 0.9 * y_final

            try:
                if y_final > 0:
                    idx10 = np.where(y >= y10)[0][0]
                    idx90 = np.where(y >= y90)[0][0]
                else:
                    idx10 = np.where(y <= y10)[0][0]
                    idx90 = np.where(y <= y90)[0][0]

                if idx90 >= idx10:
                    metrics["rise_time"] = float(t[idx90] - t[idx10])
            except IndexError:
                metrics["rise_time"] = None

        # Settling Time: last time outside tolerance band
        if abs(y_final) > 1e-12:
            band = tol * abs(y_final)
            lower = y_final - band
            upper = y_final + band

            outside = np.where((y < lower) | (y > upper))[0]
            if len(outside) == 0:
                metrics["settling_time"] = 0.0
            elif outside[-1] < len(t) - 1:
                metrics["settling_time"] = float(t[outside[-1] + 1])
            else:
                metrics["settling_time"] = None

        return metrics

    def _compute_step_metrics_safe(self, t, y, tol=0.02):
        """Return step metrics safely without breaking the whole numerical solve."""
        try:
            return self._compute_step_metrics(t, y, tol=tol)
        except Exception:
            return {
                "final_value": None,
                "peak_value": None,
                "overshoot_percent": None,
                "rise_time": None,
                "settling_time": None,
                "settling_band_percent": float(tol * 100.0),
            }

    def solve_response(self, tf_sympy, response_type='step', t_end=0):
        if response_type not in ('step', 'impulse'):
            raise ValueError("response_type must be either 'step' or 'impulse'")

        # 1) Convert to a clean rational form
        tf_clean = sympy.together(sympy.cancel(tf_sympy))
        self._validate_numeric_expr(tf_clean)

        num_expr, den_expr = sympy.fraction(tf_clean)

        # 2) Build numerator and denominator polynomials
        try:
            num_poly = sympy.Poly(num_expr, self.s)
            den_poly = sympy.Poly(den_expr, self.s)
        except Exception as e:
            raise ValueError(f"Failed to construct a polynomial in variable {self.s}: {e}")

        # 3) Basic validation
        if den_expr == 0 or den_poly.is_zero:
            raise ValueError("The denominator is zero; a valid transfer function cannot be formed")

        if num_poly.is_zero:
            t_end_used = float(t_end) if t_end and t_end > 0 else 10.0
            n_points = 500
            t = np.linspace(0.0, t_end_used, n_points)
            y = np.zeros_like(t)
            return {
                't': t,
                'y': y,
                't_end_used': t_end_used,
                'step_metrics': self._compute_step_metrics_safe(t, y) if response_type == 'step' else None,
            }

        deg_num = num_poly.degree()
        deg_den = den_poly.degree()

        if deg_num > deg_den:
            raise ValueError(
                "Improper transfer functions are not suitable for direct numerical "
                "step/impulse simulation. Please reduce the model or separate the polynomial part first"
            )

        # 4) Convert coefficients to floats
        try:
            num_coeffs = [float(sympy.N(c)) for c in num_poly.all_coeffs()]
            den_coeffs = [float(sympy.N(c)) for c in den_poly.all_coeffs()]
        except Exception as e:
            raise ValueError(f"Failed to convert polynomial coefficients to numeric values: {e}")

        num_coeffs = self._trim_leading_zeros(num_coeffs)
        den_coeffs = self._trim_leading_zeros(den_coeffs)

        if not den_coeffs or abs(den_coeffs[0]) < 1e-14:
            raise ValueError("The leading denominator coefficient is invalid")

        # 5) Build LTI system
        try:
            sys = signal.TransferFunction(num_coeffs, den_coeffs)
        except Exception as e:
            raise ValueError(f"Failed to construct the LTI system: {e}")

        # 6) Determine simulation time horizon
        if t_end is None or t_end <= 0:
            t_end_used = self._auto_time_scale(sys)
        else:
            t_end_used = float(t_end)

        # 7) Generate time vector
        n_points = int(np.clip(max(500, 80 * t_end_used), 500, 4000))
        t = np.linspace(0.0, t_end_used, n_points)

        # 8) Compute response
        try:
            if response_type == 'impulse':
                t, y = signal.impulse(sys, T=t)
            else:
                t, y = signal.step(sys, T=t)
        except Exception as e:
            raise ValueError(f"Failed to compute the {response_type} response: {e}")

        # 9) Validate outputs
        t = np.asarray(t, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
        self._validate_response_arrays(t, y)

        # 10) Optional step metrics
        step_metrics = None
        if response_type == 'step':
            step_metrics = self._compute_step_metrics_safe(t, y, tol=0.02)

        return {
            't': t,
            'y': y,
            't_end_used': float(t_end_used),
            'step_metrics': step_metrics,
        }