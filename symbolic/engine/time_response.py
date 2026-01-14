# ==================================================
# TF7 — Time Response Engine
# ==================================================
# Symbolic impulse / step response using inverse Laplace
#
# Design principles:
# - Symbolic-first
# - Backend only (no UI, no plotting)
# - No numeric sampling by default
# - Clear separation from s-domain engines
# ==================================================

from multiprocessing import Process, Queue
from queue import Empty
from sympy.integrals import inverse_laplace_transform
from sympy import simplify, symbols
from .control_engine import SymbolicEngineError


def _inverse_laplace_worker(q, Ys, s, t):
    """
    Worker process for inverse Laplace.
    Runs in a separate process to allow timeout.
    """
    try:
        yt = inverse_laplace_transform(
            Ys,     # ❗ do NOT simplify before inversion
            s,
            t,
            noconds=True
        )
        q.put(("ok", simplify(yt)))
    except Exception as e:
        q.put(("error", str(e)))


class TimeResponseEngine:
    """
    TimeResponseEngine (TF7)

    Computes symbolic time-domain responses from a transfer function G(s)
    using inverse Laplace transform.

    Supported response types:
    - 'impulse' : impulse response
    - 'step'    : step response
    """

    def __init__(self, s=None, t=None):
        """
        Parameters
        ----------
        s : sympy.Symbol, optional
            Laplace-domain symbol. MUST match upstream engine.
        t : sympy.Symbol, optional
            Time-domain symbol.
        """
        self.s = s if s is not None else symbols("s")
        self.t = t if t is not None else symbols("t", positive=True)

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def compute(self, Gs, response_type: str):
        """
        Compute symbolic time response y(t).
        """

        if Gs is None:
            raise SymbolicEngineError(
                "Transfer function G(s) is None",
                stage="TF7"
            )

        # ----------------------------------------------
        # Apply response semantics
        # ----------------------------------------------
        if response_type == "impulse":
            Ys = self._apply_impulse(Gs)

        elif response_type == "step":
            Ys = self._apply_step(Gs)

        else:
            raise SymbolicEngineError(
                f"Unsupported response type: {response_type}",
                stage="TF7"
            )

        # ----------------------------------------------
        # Inverse Laplace (with timeout)
        # ----------------------------------------------
        return self._inverse_laplace_with_timeout(Ys, timeout=3.0)

    # --------------------------------------------------
    # Response semantics
    # --------------------------------------------------

    def _apply_impulse(self, Gs):
        """
        Laplace{δ(t)} = 1  ⇒  Y(s) = G(s)
        """
        return simplify(Gs)

    def _apply_step(self, Gs):
        """
        Laplace{u(t)} = 1/s  ⇒  Y(s) = G(s)/s
        """
        return simplify(Gs / self.s)

    # --------------------------------------------------
    # Inverse Laplace core
    # --------------------------------------------------

    def _inverse_laplace_with_timeout(self, Ys, timeout=3.0):
        """
        Perform inverse Laplace transform with timeout (seconds).
        """

        q = Queue()
        p = Process(
            target=_inverse_laplace_worker,
            args=(q, Ys, self.s, self.t)
        )

        p.start()
        p.join(timeout)

        # ---- TIMEOUT ----
        if p.is_alive():
            p.terminate()
            p.join()
            raise SymbolicEngineError(
                "Inverse Laplace transform timed out.\n"
                "The transfer function may be too complex for symbolic inversion.",
                stage="TF7"
            )

        # ---- SAFE QUEUE READ ----
        try:
            status, payload = q.get(timeout=0.2)
        except Empty:
            raise SymbolicEngineError(
                "Inverse Laplace worker terminated without returning a result.",
                stage="TF7"
            )

        if status == "ok":
            return payload
        else:
            raise SymbolicEngineError(
                "Inverse Laplace transform failed.\n"
                f"Details: {payload}",
                stage="TF7"
            )
