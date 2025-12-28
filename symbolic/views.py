from django.shortcuts import render

from .engine.control_engine import (
    SymbolicControlEngine,
    SymbolicEngineError,
    BlockDiagramEngine,
    block_graph_to_mermaid,
)


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
    - tf      : compute symbolic transfer function
    - block   : generate structural block diagram
    """

    # --------------------------------------------------
    # Default context (สำคัญ: ป้องกันค่าหาย)
    # --------------------------------------------------
    context = {
        # ===== must be consistent with DEFAULT_EQUATIONS =====
        "num_vars": 3,
        "num_forward": 2,
        "num_backward": 2,

        "equations": DEFAULT_EQUATIONS,
        "forward_gain": "",
        "backward_gain": "",

        "response_type": "nilt",
        "dt": "0.01",
        "points": 500,

        "variables": "",
        "result": None,
        "error": None,

        # LaTeX outputs
        "latex_equations": [],
        "latex_tf": None,

        # Block diagram (Mermaid)
        "mermaid": None,
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

            context["response_type"] = request.POST.get(
                "response_type", "nilt"
            )
            context["dt"] = request.POST.get("dt", context["dt"])
            context["points"] = request.POST.get(
                "points", context["points"]
            )

            action = request.POST.get("action", "tf")

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
            # Initialize symbolic engine (ใช้ร่วมทุก action)
            # ----------------------------------------------
            engine = SymbolicControlEngine()

            engine.init_variables(context["num_vars"])
            engine.init_forward_gains(context["num_forward"])
            engine.init_backward_gains(context["num_backward"])

            engine.load_equations(equation_list)

            # parse gain values (comma-separated)
            g_vals = [
                g.strip()
                for g in context["forward_gain"].split(",")
                if g.strip()
            ]
            h_vals = [
                h.strip()
                for h in context["backward_gain"].split(",")
                if h.strip()
            ]

            engine.load_forward_gain_values(g_vals)
            engine.load_backward_gain_values(h_vals)

            # ==================================================
            # ACTION: Transfer Function
            # ==================================================
            if action == "tf":
                engine.compute_symbolic_transfer_function()
                engine.substitute_gains()

                context["latex_equations"] = engine.get_latex_equations()
                context["latex_tf"] = engine.get_latex_transfer_function()

                context["result"] = (
                    "Overall transfer function computed successfully\n"
                    "-----------------------------------------------\n"
                    f"Number of variables      : {context['num_vars']}\n"
                    f"Number of forward gains  : {context['num_forward']}\n"
                    f"Number of backward gains : {context['num_backward']}\n"
                )

            # ==================================================
            # ACTION: Block Diagram
            # ==================================================
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
