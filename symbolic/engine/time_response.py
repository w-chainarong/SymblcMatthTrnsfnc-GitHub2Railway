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

import multiprocessing
import platform
import os
from multiprocessing import Process, Queue

# บังคับใช้ 'spawn' สำหรับระบบ Linux (Railway) 
# เพื่อป้องกันปัญหา Fork Safety และ Database Connection Error
if platform.system() != 'Windows':
    try:
        # โหมด 'spawn' จะทำงานเสถียรและเหมือนกับบน Windows (PC) มากที่สุด
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # ป้องกัน Error ในกรณีที่ Python มีการตั้งค่าไปแล้วใน Module อื่น
        pass

from sympy.integrals import inverse_laplace_transform
from sympy import simplify, symbols
from .control_engine import SymbolicEngineError

# ==================================================
# TF7 — Time Response Engine
# ==================================================

def _inverse_laplace_worker(q, Ys, s, t):
    """
    Worker process for inverse Laplace.
    This runs in a separate process to allow timeout.
    """
    try:
        yt = inverse_laplace_transform(
            simplify(Ys),
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
            Laplace-domain symbol. If provided, MUST be the same symbol
            used to construct G(s). This is critical for correctness.
        t : sympy.Symbol, optional
            Time-domain symbol.
        """
        # IMPORTANT:
        # Use the SAME Laplace symbol as the upstream control engine
        self.s = s if s is not None else symbols("s")
        self.t = t if t is not None else symbols("t", positive=True)

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def compute(self, Gs, response_type: str):
        """
        Compute symbolic time response y(t).

        Parameters
        ----------
        Gs : sympy.Expr
            Transfer function G(s)
        response_type : str
            'impulse' or 'step'

        Returns
        -------
        sympy.Expr
            Symbolic time response y(t)
        """

        if Gs is None:
            raise SymbolicEngineError(
                "Transfer function G(s) is None",
                stage="time response engine"
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
                stage="time response engine"
            )

        # ----------------------------------------------
        # Inverse Laplace
        # ----------------------------------------------
        return self._inverse_laplace_with_timeout(Ys, timeout=3.0)

    # --------------------------------------------------
    # Response semantics
    # --------------------------------------------------

    def _apply_impulse(self, Gs):
        """
        Impulse response semantics.

        Laplace{δ(t)} = 1
        ⇒ Y(s) = G(s)
        """
        return simplify(Gs)

    def _apply_step(self, Gs):
        """
        Step response semantics.

        Laplace{u(t)} = 1/s
        ⇒ Y(s) = G(s) / s
        """
        return simplify(Gs / self.s)

    # --------------------------------------------------
    # Inverse Laplace core
    # --------------------------------------------------
    def _inverse_laplace(self, Ys):
        """
        Perform inverse Laplace transform (safe symbolic mode).
        """

        try:
            yt = inverse_laplace_transform(
                Ys,          # ❗ ไม่จำเป็นต้อง simplify ก่อน
                self.s,
                self.t,
                noconds=True
            )

            return simplify(yt)

        except Exception as e:
            raise SymbolicEngineError(
                "Inverse Laplace transform failed.\n"
                "The transfer function may be too complex for symbolic inversion.\n"
                f"Details: {e}",
                stage="time response engine"
        )

    def _inverse_laplace_with_timeout(self, Ys, timeout=3.0):
        """
        Perform inverse Laplace transform with timeout (seconds).
        ปรับปรุงเพื่อรองรับการรันบน Railway/Cloud
        """

        q = Queue()
        p = Process(
            target=_inverse_laplace_worker,
            args=(q, Ys, self.s, self.t)
        )

        p.start()
        p.join(timeout)

        # ---- 1. จัดการกรณีใช้เวลานานเกินกำหนด (TIMEOUT) ----
        if p.is_alive():
            p.terminate()  # สั่งหยุดการทำงานทันที
            p.join()       # รอให้คืนทรัพยากร (RAM 8GB) กลับสู่ระบบ
            raise SymbolicEngineError(
                "Inverse Laplace transform timed out.\n"
                "The transfer function appears too complex for symbolic inversion. " \
                "Please use the numerical solution instead.",
                stage="time response engine"
            )

        # ---- 2. ดึงข้อมูลจาก Queue อย่างปลอดภัย (ป้องกันแอปค้าง) ----
        try:
            # ใช้ block=False เพื่อไม่ให้โปรแกรมหลักค้าง 
            # ในกรณีที่ Process ลูกแครชกะทันหันก่อนส่งข้อมูลลง Queue
            status, payload = q.get(block=False)
        except:
            # ถ้าดึงข้อมูลไม่ได้ แสดงว่า Worker process ตาย (เช่นโดนระบบ Kill เพราะ RAM พุ่ง)
            raise SymbolicEngineError(
                "Calculation process failed or was interrupted unexpectedly.",
                stage="time response engine"
            )

        # ---- 3. ตรวจสอบผลลัพธ์จาก Worker ----
        if status == "ok":
            return payload
        else:
            raise SymbolicEngineError(
                f"Inverse Laplace transform failed.\nDetails: {payload}",
                stage="time response engine"
            )


