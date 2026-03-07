from django.shortcuts import render
import textwrap

from .engine.control_engine import (
    SymbolicControlEngine,
    SymbolicEngineError,
    BlockDiagramEngine,
    block_graph_to_mermaid,
)

from .engine.zpk_minreal import ZPKMinrealEngine
from .engine.time_response import TimeResponseEngine
from .engine.numerical_engine import NumericalEngine

from sympy import latex, together, expand


# ==================================================
# Default symbolic system (DEMO / BASELINE)
# ==================================================
DEFAULT_EQUATIONS = """\
x1 - R + x3
x2 - x1*G1 + x3*H2*H1
x3 - x2*G2
"""

def clean_expression(expr, tol=1e-12):
    expr = expand(expr)

    # ลบ tiny coefficient ระดับ atom
    expr = expr.xreplace({
        c: 0 for c in expr.atoms()
        if c.is_Number and abs(complex(c)) < tol
    })

    expr = expr.evalf(chop=True)

    real_part, imag_part = expr.as_real_imag()

    try:
        if abs(float(imag_part)) < tol:
            return real_part
    except:
        pass

    return expr

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
    "warning": None,

    # LaTeX outputs
    "latex_equations": [],
    "latex_tf": None,
    "latex_algebraic_tf": None,
    "latex_tf_structural": None,
    "latex_tf_used": None,
    "latex_time_response": None,

    # Titles and display control
    "context_action": None,
    "tf_display_title": None,
    "tf_structural_title": "Symbolic Overall Transfer Function",
    "show_algebraic_tf": False,

    # Diagram and numerical outputs
    "mermaid": None,
    "numerical_data": None,
    "step_metrics": None,
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
            context["context_action"] = action
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
            # แสดง System of Equations เสมอสำหรับทุก action
            context["latex_equations"] = engine.get_latex_equations()

            # คำนวณ Symbolic Overall Transfer Function เสมอสำหรับทุก action
            tf_engine = SymbolicControlEngine()
            tf_engine.init_variables(context["num_vars"])
            tf_engine.init_forward_gains(context["num_forward"])
            tf_engine.init_backward_gains(context["num_backward"])
            tf_engine.load_equations(equation_list)
            tf_engine.compute_symbolic_transfer_function()
            context["latex_tf_structural"] = tf_engine.get_latex_transfer_function()

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
            # ----------------------------------------------
            # Reset action-specific outputs
            # ----------------------------------------------
            context["latex_tf"] = None
            context["latex_algebraic_tf"] = None
            context["show_algebraic_tf"] = False
            context["mermaid"] = None
            context["warning"] = None
            context["step_metrics"] = None
            context["latex_tf_used"] = None
            context["latex_time_response"] = None
            context["numerical_data"] = None

            # ==================================================
            # ACTION: TF
            # ==================================================
            if action == "tf":
                engine.compute_symbolic_transfer_function()
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

                context["latex_algebraic_tf"] = engine.get_latex_algebraic_transfer_function()
                context["show_algebraic_tf"] = True

                # ----------------------------------------------
                # STEP 5: ZPK (Minreal)
                # ----------------------------------------------
                if tf_mode == "zpk_minreal":
                    tf_expr = engine.get_transfer_function_expr()
                    s = engine.s

                    zpk_engine = ZPKMinrealEngine(tf_expr, s)
                    zpk_engine.compute_zpk()
                    zpk_engine.minreal()

                    context["latex_tf"] = zpk_engine.to_latex()
                    context["tf_display_title"] = "Overall Transfer Function (Laplace Domain, ZPK / Minreal)"
                    context["result"] = (
                        "ZPK (Minreal) Laplace-domain transfer function\n"
                        "---------------------------------------------\n"
                        + zpk_engine.to_matlab_like()
                    )

                    return render(
                        request,
                        "symbolic/runtime_gui.html",
                        context
                    )

                # ----------------------------------------------
                # STEP 4: Symbolic Laplace modes
                # ----------------------------------------------
                context["latex_tf"] = engine.format_transfer_function(tf_mode)

                if tf_mode == "factorized":
                    context["tf_display_title"] = "Overall Transfer Function (Laplace Domain, Symbolic Factorized)"
                    context["result"] = (
                        "Laplace-domain overall transfer function\n"
                        "----------------------------------------\n"
                        "Display mode: factorized\n"
                        "Gain expressions G(s), H(s) substituted.\n"
                    )
                else:
                    context["tf_display_title"] = "Overall Transfer Function (Laplace Domain, Raw Symbolic)"
                    context["result"] = (
                        "Laplace-domain overall transfer function\n"
                        "----------------------------------------\n"
                        "Display mode: raw\n"
                        "Gain expressions G(s), H(s) substituted.\n"
                    )
            elif action == "block":
                bd_engine = BlockDiagramEngine(engine)
                graph = bd_engine.build_block_graph()

                context["mermaid"] = block_graph_to_mermaid(graph)
                context["result"] = (
                    "Block diagram generated successfully\n"
                    "-----------------------------------\n"
                    f"Number of nodes : {len(graph['nodes'])}\n"
                    f"Number of edges : {len(graph['edges'])}\n"
                )
            # ==================================================
            # ACTION: Inverse Laplace (ℒ⁻¹)
            # ==================================================
            elif action == "inv_laplace":
                response_type = request.POST.get("response_type", "impulse")
                solution_type = request.POST.get("solution_type", "symbolic")
                
                # --- [เพิ่ม] ดึงค่า Numerical Parameters ---
                is_auto_scale = request.POST.get("auto_scale") == "on"
                try:
                    t_end_val = float(request.POST.get("t_end", 10.0))
                except:
                    t_end_val = 10.0

                # 1) Load gains
                engine.load_forward_gain_expressions(g_vals)
                engine.load_backward_gain_expressions(h_vals)

                # 2) Compute Laplace-domain transfer function
                engine.compute_symbolic_transfer_function()
                engine.substitute_gains()

                # 3) Select G(s)
                if tf_mode == "raw":
                    Gs = engine.get_transfer_function_expr()
                elif tf_mode == "factorized":
                    Gs = engine.get_transfer_function_expr().factor()
                elif tf_mode == "zpk_minreal":
                    tf_expr = engine.get_transfer_function_expr()
                    s_sym = engine.s
                    zpk_engine = ZPKMinrealEngine(tf_expr, s_sym)
                    zpk_engine.compute_zpk()
                    zpk_engine.minreal()
                    Gs = zpk_engine.get_minreal_expr()
                else:
                    raise SymbolicEngineError("Unsupported TF mode", stage="TF7")
                Gs = clean_expression(Gs)
                Gs = Gs.nsimplify()
                Gs = together(Gs)
                context["latex_tf_used"] = latex(Gs)
                s = engine.s
                # --- [แยกฝั่งการทำงาน] ---
                if solution_type == "numeric":
                    # --------------------------------------------------
                    # โหมด Numerical
                    # --------------------------------------------------
                    num_eng = NumericalEngine(s)
                    t_limit = 0 if is_auto_scale else t_end_val
                    res = num_eng.solve_response(Gs, response_type=response_type, t_end=t_limit)

                    context["numerical_data"] = {
                        "t": res["t"].tolist(),
                        "y": res["y"].tolist(),
                        "label": f"{response_type.capitalize()} Response",
                        "t_used": res["t_end_used"],
                        "step_metrics": res.get("step_metrics"),
                    }
                    context["latex_time_response"] = r"\text{Numerical results generated (See graph below)}"
                    context["result"] = f"Response: {response_type}\nSolution: Numerical"
                    context["step_metrics"] = res.get("step_metrics")
                else:
                    # --------------------------------------------------
                    # โหมด Symbolic (ย้าย Logic เดิมมาไว้ที่นี่)
                    # --------------------------------------------------
                    context["warning"] = None
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
                    except:
                        pass

                    tr_engine = TimeResponseEngine(s=s)
                    yt = tr_engine.compute(Gs=Gs, response_type=response_type)
                    
                    context["latex_time_response"] = latex(yt)
                    context["result"] = f"Response type : {response_type}\nSolution type : symbolic"

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
