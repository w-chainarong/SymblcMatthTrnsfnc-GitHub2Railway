from django.shortcuts import render
import textwrap

from .engine.control_engine import (
    SymbolicControlEngine,
    SymbolicEngineError,
    BlockDiagramEngine,
    block_graph_to_mermaid,
    validate_block_graph,
    TopologyError,
)

# STEP 5: ZPK (Minreal)
from .engine.zpk_minreal import ZPKMinrealEngine

from .engine.time_response import TimeResponseEngine
from sympy import latex
from sympy import pretty

# ==================================================
# Default symbolic system (DEMO / BASELINE)
# ==================================================
DEFAULT_EQUATIONS = """\
x1 - R + x3
x2 - x1*G1 + x3*H2*H1
x3 - x2*G2
"""


def runtime_gui(request):
    """
    Runtime symbolic control-system GUI

    Supported actions:
    - tf        : structural symbolic transfer function (NO gain substitution)
    - laplace   : Laplace-domain transfer function (WITH gain substitution)
    - block     : generate structural block diagram
    """

    # --------------------------------------------------
    # Default context (สำคัญ: ป้องกันค่าหาย)
    # --------------------------------------------------
    context = {
        "num_vars": 3,
        "num_forward": 2,
        "num_backward": 2,
        "tf_mode": "raw",

        "equations": DEFAULT_EQUATIONS,
        "forward_gain": textwrap.dedent("""\
            (2*s + 1)/(s + 2)
            1/(s*(s**2 + 2*s + 2))
        """),

        "backward_gain": textwrap.dedent("""\
            2*s
            (s - 1)/(s + 1)
        """),
        "variables": "",
        "result": None,
        "error": None,

        # LaTeX outputs
        "latex_equations": [],
        "latex_tf": None,

        # Block diagram (Mermaid)
        "mermaid": None,
        "warning": None,
    }

    # --------------------------------------------------
    # POST handling
    # --------------------------------------------------
    if request.method == "POST":
        try:
            # ----------------------------------------------
            # Read scalar inputs
            # ----------------------------------------------
            context["num_vars"] = int(
                request.POST.get("num_vars", context["num_vars"])
            )
            context["num_forward"] = int(
                request.POST.get("num_forward", context["num_forward"])
            )
            context["num_backward"] = int(
                request.POST.get("num_backward", context["num_backward"])
            )

            context["forward_gain"] = request.POST.get("forward_gain", "")
            context["backward_gain"] = request.POST.get("backward_gain", "")

            action = request.POST.get("action", "tf")
            tf_mode = request.POST.get("tf_mode", "raw")

            # ----------------------------------------------
            # Normalize tf_mode (สำคัญมาก)
            # ----------------------------------------------
            if tf_mode == "zpk":
                tf_mode = "zpk_minreal"

            ALLOWED_TF_MODES = {"raw", "factorized", "zpk_minreal"}
            if tf_mode not in ALLOWED_TF_MODES:
                tf_mode = "raw"

            context["tf_mode"] = tf_mode

            # ----------------------------------------------
            # Read multi-line equations (implicit form)
            # ----------------------------------------------
            raw_equations = request.POST.get("equations", "").strip()
            if raw_equations:
                context["equations"] = raw_equations
            else:
                context["equations"] = DEFAULT_EQUATIONS

            equation_list = [
                line.strip()
                for line in context["equations"].splitlines()
                if line.strip()
            ]

            # ----------------------------------------------
            # Variable display (x1, x2, ...)
            # ----------------------------------------------
            n = context["num_vars"]
            context["variables"] = ", ".join(
                [f"x{i+1}" for i in range(n)]
            )

            # ----------------------------------------------
            # Initialize symbolic engine (STEP 1–4)
            # ----------------------------------------------
            engine = SymbolicControlEngine()

            engine.init_variables(context["num_vars"])
            engine.init_forward_gains(context["num_forward"])
            engine.init_backward_gains(context["num_backward"])

            engine.load_equations(equation_list)

            # ----------------------------------------------
            # Parse multiline gain expressions
            # ----------------------------------------------
            g_vals = [
                line.strip()
                for line in context["forward_gain"].splitlines()
                if line.strip()
            ]

            h_vals = [
                line.strip()
                for line in context["backward_gain"].splitlines()
                if line.strip()
            ]

            # ==================================================
            # ACTION: TF
            # ==================================================
            if action == "tf":
                engine.compute_symbolic_transfer_function()
                context["latex_equations"] = engine.get_latex_equations()
            # ----------------------------------------------
            # STEP 5: ZPK (Minreal) — TEMPORARILY DISABLED
            # ----------------------------------------------
            # if tf_mode == "zpk_minreal":
            #     tf_expr = engine.get_transfer_function_expr()
            #     s = engine.s
            #
            #     zpk_engine = ZPKMinrealEngine(tf_expr, s)
            #     zpk_engine.compute_zpk()
            #     zpk_engine.minreal()
            #
            #     context["latex_tf"] = zpk_engine.to_latex()
            #     context["result"] = (
            #         "ZPK (Minreal) overall transfer function\n"
            #         "--------------------------------------\n"
            #         + zpk_engine.to_matlab_like()
            #     )
            #
            #     return render(
            #         request,
            #         "symbolic/runtime_gui.html",
            #         context
            #     )

                # ----------------------------------------------
                # STEP 4: Symbolic modes ONLY
                # ----------------------------------------------
                context["latex_tf"] = engine.get_latex_transfer_function()
                context["result"] = (
                    "Structural (symbolic) overall transfer function\n"
                    "------------------------------------------------\n"
                    "Gain expressions G(s), H(s) are NOT substituted.\n"
                )

            # ==================================================
            # ACTION: Laplace (ℒ)
            # ==================================================
            elif action == "laplace":
                engine.load_forward_gain_expressions(g_vals)
                engine.load_backward_gain_expressions(h_vals)

                engine.compute_symbolic_transfer_function()
                engine.substitute_gains()

                context["latex_equations"] = engine.get_latex_equations()
                context["latex_algebraic_tf"] = engine.get_latex_algebraic_transfer_function()
                # ----------------------------------------------
                # STEP 5: ZPK (Minreal) — TERMINAL MODE
                # ----------------------------------------------
                if tf_mode == "zpk_minreal":
                    tf_expr = engine.get_transfer_function_expr()
                    s = engine.s

                    zpk_engine = ZPKMinrealEngine(tf_expr, s)
                    zpk_engine.compute_zpk()
                    zpk_engine.minreal()

                    context["latex_tf"] = zpk_engine.to_latex()
                    context["result"] = (
                        "ZPK (Minreal) Laplace-domain transfer function\n"
                        "---------------------------------------------\n"
                        + zpk_engine.to_matlab_like()
                    )

                    # 🔴 สำคัญที่สุด: ตัด flow ตรงนี้
                    return render(
                        request,
                        "symbolic/runtime_gui.html",
                        context
                    )

                # ----------------------------------------------
                # STEP 4: Symbolic Laplace modes
                # ----------------------------------------------
                context["latex_tf"] = engine.format_transfer_function(tf_mode)
                context["result"] = (
                    "Laplace-domain overall transfer function\n"
                    "----------------------------------------\n"
                    "Gain expressions G(s), H(s) substituted.\n"
                )
            # ==================================================
            # ACTION: Block Diagram
            # ==================================================
            elif action == "block":
                bd_engine = BlockDiagramEngine(engine)
                graph = bd_engine.build_block_graph()
                # --- FINAL-STRICT validation (simple) ---
                try:
                    validate_block_graph(graph)
                    topology_status = "FINAL-STRICT VALID"
                except TopologyError as e:
                    topology_status = f"INVALID: {e}"

                context["mermaid"] = block_graph_to_mermaid(graph)
                if topology_status == "FINAL-STRICT VALID":
                    context["result"] = (
                        "Block diagram generated successfully\n"
                        "-----------------------------------\n"
                        f"Number of nodes : {len(graph['nodes'])}\n"
                        f"Number of edges : {len(graph['edges'])}\n\n"
                        "Topology status : FINAL-STRICT VALID\n\n"
                        "Validation checklist:\n"
                        "  ✔ Summing junctions have ≥ 2 inputs and 1 output\n"
                        "  ✔ Take-off points have exactly 1 input and ≥ 2 outputs\n"
                        "  ✔ No self-loop or algebraic loop detected\n"
                        "  ✔ Output Y has exactly 1 input and no outgoing edges\n"
                    )
                else:
                    context["result"] = (
                        "Block diagram generated successfully\n"
                        "-----------------------------------\n"
                        f"Number of nodes : {len(graph['nodes'])}\n"
                        f"Number of edges : {len(graph['edges'])}\n\n"
                        f"Topology status : {topology_status}\n"
                    )
            # ==================================================
            # ACTION: Inverse Laplace (ℒ⁻¹)
            # ==================================================
            elif action == "inv_laplace":
                response_type = request.POST.get("response_type", "impulse")
                solution_type = request.POST.get("solution_type", "symbolic")

                if solution_type != "symbolic":
                    raise SymbolicEngineError(
                        "Only symbolic inverse Laplace is supported.",
                        stage="TF7"
                    )

                # 1) Load gains
                engine.load_forward_gain_expressions(g_vals)
                engine.load_backward_gain_expressions(h_vals)

                # 2) Compute Laplace-domain transfer function
                engine.compute_symbolic_transfer_function()
                engine.substitute_gains()

                # 3) Select G(s) according to tf_mode
                if tf_mode == "raw":
                    Gs = engine.get_transfer_function_expr()

                elif tf_mode == "factorized":
                    Gs = engine.get_transfer_function_expr().factor()

                elif tf_mode == "zpk_minreal":
                    tf_expr = engine.get_transfer_function_expr()
                    s = engine.s

                    zpk_engine = ZPKMinrealEngine(tf_expr, s)
                    zpk_engine.compute_zpk()
                    zpk_engine.minreal()

                    Gs = zpk_engine.get_minreal_expr()

                else:
                    raise SymbolicEngineError(
                        "Unsupported TF mode for inverse Laplace",
                        stage="TF7"
                    )
                context["latex_tf_used"] = latex(Gs)
                # 4) WARNING: improper transfer function (SAFE POSITION)
                context["warning"] = None
                s = engine.s

                try:
                    num, den = Gs.as_numer_denom()
                    deg_num = num.as_poly(s).degree()
                    deg_den = den.as_poly(s).degree()

                    if deg_num > deg_den:
                        context["warning"] = (
                            "⚠️ Warning: Improper transfer function detected.\n"
                            f"deg(numerator) = {deg_num} > deg(denominator) = {deg_den}\n"
                            "Only the regular (proper) time-domain response will be shown.\n"
                            "Distribution terms (e.g. DiracDelta) are omitted."
                        )
                except Exception:
                    # Degree detection failed → do not block computation
                    pass
                # --------------------------------------------------
                # STEP 4: Time-domain response via TF7
                # --------------------------------------------------
                tr_engine = TimeResponseEngine(
                    s=engine.s   # สำคัญ: ใช้ symbol เดียวกับ ControlEngine
                )

                yt = tr_engine.compute(
                    Gs=Gs,
                    response_type=response_type   # "impulse" หรือ "step"
                )

                context["latex_time_response"] = latex(yt)

                context["result"] = (
                    f"Response type : {response_type}\n"
                    "Solution type : symbolic\n"
                )


        # ==================================================
        # Controlled symbolic error
        # ==================================================
        except SymbolicEngineError as e:
            context["error"] = str(e)

        # ==================================================
        # Unexpected system / programming error
        # ==================================================
        except Exception as e:
            context["error"] = (
                "Unexpected internal error.\n"
                "Please check input or contact administrator.\n\n"
                f"Details: {e}"
            )

    return render(request, "symbolic/runtime_gui.html", context)
