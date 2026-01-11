from sympy import symbols, Eq, solve, simplify, latex, Symbol
from sympy.parsing.sympy_parser import parse_expr
from sympy.core.sympify import SympifyError
from sympy import expand
from sympy import cancel, factor, fraction, LC

import re


# ==================================================
# Custom Engine Error
# ==================================================

class SymbolicEngineError(Exception):
    """
    Custom exception for symbolic control engine.
    Wraps native SymPy errors with semantic context.
    """
    def __init__(self, message: str, stage: str | None = None):
        self.stage = stage
        super().__init__(message)

    def __str__(self):
        if self.stage:
            return f"[{self.stage}] {super().__str__()}"
        return super().__str__()


# ==================================================
# Symbolic Control Engine
# ==================================================

class SymbolicControlEngine:
    """
    Symbolic Control-System Engine (Django-ready)

    - Stateless
    - Semantic whitelist validation
    - Line-aware diagnostics
    - One instance per request
    """

    # --------------------------------------------------
    # Constructor
    # --------------------------------------------------

    def __init__(self):
        self.num_vars = 0
        self.num_forward = 0
        self.num_backward = 0

        # symbols
        self.x = []     # x1, x2, ...
        self.X = []

        self.g = []     # G1, G2, ...
        self.h = []     # H1, H2, ...

        self.R = symbols('R')
        self.s = Symbol('s')

        # expressions
        self.equations = []
        self.forward_gain_values = {}     # {Gi: expr}
        self.backward_gain_values = {}    # {Hi: expr}

        # results
        self.symbolic_tf = None
        self.s_domain_tf = None

    # --------------------------------------------------
    # Initialization
    # --------------------------------------------------

    def init_variables(self, num_vars: int):
        if num_vars <= 0:
            raise SymbolicEngineError(
                "Number of variables must be positive",
                stage="init_variables"
            )
        self.num_vars = num_vars
        self.x = symbols(f'x1:{num_vars + 1}')
        self.X = [f"x{i+1}" for i in range(num_vars)]

    def init_forward_gains(self, num_forward: int):
        if num_forward < 0:
            raise SymbolicEngineError(
                "Number of forward gains cannot be negative",
                stage="init_forward_gains"
            )
        self.num_forward = num_forward
        self.g = symbols(f'G1:{num_forward + 1}')

    def init_backward_gains(self, num_backward: int):
        if num_backward < 0:
            raise SymbolicEngineError(
                "Number of backward gains cannot be negative",
                stage="init_backward_gains"
            )
        self.num_backward = num_backward
        self.h = symbols(f'H1:{num_backward + 1}')

    # --------------------------------------------------
    # Semantic whitelist
    # --------------------------------------------------

    def _allowed_symbols(self) -> set:
        return set(self.x) | set(self.g) | set(self.h) | {self.R, self.s}

    def _validate_symbols(self, expr, stage: str):
        illegal = expr.free_symbols - self._allowed_symbols()
        if illegal:
            names = ", ".join(sorted(str(s) for s in illegal))
            raise SymbolicEngineError(
                f"Illegal symbol(s) detected: {names}",
                stage=stage
            )

    def _validate_gain_symbols(self, expr, stage: str):
        """
        Gain expressions must be Laplace-domain only:
        allowed symbol: s
        """
        illegal = expr.free_symbols - {self.s}
        if illegal:
            names = ", ".join(sorted(str(s) for s in illegal))
            raise SymbolicEngineError(
                f"Illegal symbol(s) in gain expression: {names}",
                stage=stage
            )


    # --------------------------------------------------
    # Input loading (LINE-AWARE)
    # --------------------------------------------------

    def load_equations(self, equations: list[str]):
        self.equations = []

        if len(equations) != self.num_vars:
            raise SymbolicEngineError(
                "Number of equations must equal number of variables",
                stage="load_equations"
            )

        for idx, eq_str in enumerate(equations, start=1):
            try:
                expr = parse_expr(eq_str)
                self._validate_symbols(expr, stage="parse_equations")
                self.equations.append(Eq(expr, 0))
            except (SyntaxError, SympifyError, TypeError) as e:
                raise SymbolicEngineError(
                    f"Syntax error in equation line {idx}: {e}",
                    stage="parse_equations"
                )

    # --------------------------------------------------
    # Gain loading (IDENTICAL PATTERN)
    # --------------------------------------------------

    def load_forward_gain_expressions(self, gains: list[str]):
        self.forward_gain_values = {}

        if len(gains) != self.num_forward:
            raise SymbolicEngineError(
                "Number of forward gain expressions must equal number of forward gains",
                stage="load_forward_gains"
            )

        for idx, gain_str in enumerate(gains, start=1):
            Gi = self.g[idx - 1]
            try:
                expr = parse_expr(gain_str)
                self._validate_gain_symbols(expr, stage="parse_forward_gain")
                self.forward_gain_values[Gi] = expr
            except (SyntaxError, SympifyError, TypeError) as e:
                raise SymbolicEngineError(
                    f"Syntax error in forward gain line {idx}: {e}",
                    stage="parse_forward_gain"
                )

    def load_backward_gain_expressions(self, gains: list[str]):
        self.backward_gain_values = {}

        if len(gains) != self.num_backward:
            raise SymbolicEngineError(
                "Number of backward gain expressions must equal number of backward gains",
                stage="load_backward_gains"
            )

        for idx, gain_str in enumerate(gains, start=1):
            Hi = self.h[idx - 1]
            try:
                expr = parse_expr(gain_str)
                self._validate_gain_symbols(expr, stage="parse_backward_gain")
                self.backward_gain_values[Hi] = expr
            except (SyntaxError, SympifyError, TypeError) as e:
                raise SymbolicEngineError(
                    f"Syntax error in backward gain line {idx}: {e}",
                    stage="parse_backward_gain"
                )

    # --------------------------------------------------
    # Core computation
    # --------------------------------------------------

    def compute_symbolic_transfer_function(self):
        try:
            solution = solve(self.equations, self.x)
        except Exception as e:
            raise SymbolicEngineError(
                f"Failed to solve symbolic system: {e}",
                stage="solve"
            )

        if len(solution) != self.num_vars:
            raise SymbolicEngineError(
                "Symbolic system has no unique solution",
                stage="solve"
            )

        try:
            Y = solution[self.x[-1]]
            self.algebraic_tf = Y / self.R
            self.symbolic_tf = simplify(Y / self.R)
        except Exception as e:
            raise SymbolicEngineError(
                f"Failed to compute symbolic transfer function: {e}",
                stage="simplify_tf"
            )

        return self.symbolic_tf

    def substitute_gains(self):
        if self.symbolic_tf is None:
            raise SymbolicEngineError(
                "Transfer function not computed yet",
                stage="substitute_gains"
            )

        # --- NEW: algebraic s-domain (no simplify) ---
        self.algebraic_s_domain_tf = self.algebraic_tf
        for Gi, val in self.forward_gain_values.items():
            self.algebraic_s_domain_tf = self.algebraic_s_domain_tf.subs(Gi, val)
        for Hi, val in self.backward_gain_values.items():
            self.algebraic_s_domain_tf = self.algebraic_s_domain_tf.subs(Hi, val)

        tf = self.symbolic_tf
        try:
            for Gi, val in self.forward_gain_values.items():
                tf = tf.subs(Gi, val)
            for Hi, val in self.backward_gain_values.items():
                tf = tf.subs(Hi, val)
            self.s_domain_tf = simplify(tf)
        except Exception as e:
            raise SymbolicEngineError(
                f"Failed during gain substitution: {e}",
                stage="substitute_gains"
            )

        return self.s_domain_tf

    # --------------------------------------------------
    # Output formatting
    # --------------------------------------------------

  
    def get_latex_transfer_function(self) -> str | None:
        try:
            if self.s_domain_tf is not None:
                return latex(self.s_domain_tf)
            elif self.symbolic_tf is not None:
                return latex(self.symbolic_tf)
        except Exception as e:
            raise SymbolicEngineError(
                f"Failed to generate LaTeX: {e}",
                stage="latex_export"
            )
        return None

    def get_latex_equations(self) -> list[str]:
        try:
            return [latex(eq) for eq in self.equations]
        except Exception as e:
            raise SymbolicEngineError(
                f"Failed to convert equations to LaTeX: {e}",
                stage="latex_equations"
            )

    def format_transfer_function(self, mode: str = "raw") -> str | None:
        """
        mode:
            raw     → exact symbolic (Version A)
            matlab  → cancel + normalize + factor (MATLAB-like)
        """

        # เลือก TF
        tf = self.s_domain_tf if self.s_domain_tf is not None else self.symbolic_tf
        if tf is None:
            return None

        # ---------- RAW ----------
        if mode == "raw":
            return latex(tf)

        # ---------- MATLAB-like ----------
        tf = cancel(tf)

        num, den = fraction(tf)
        lc = LC(den)
        if lc != 0:
            num = num / lc
            den = den / lc

        num = factor(num)
        den = factor(den)

        return latex(num / den)

    # --------------------------------------------------
    # Public accessor (for STEP 5: ZPK Minreal)
    # --------------------------------------------------
    def get_transfer_function_expr(self):
        """
        Return the final transfer function as a SymPy expression.

        Priority:
        1) s-domain TF (after gain substitution)
        2) symbolic TF (structural)
        """
        if self.s_domain_tf is not None:
            return self.s_domain_tf

        if self.symbolic_tf is not None:
            return self.symbolic_tf

        raise SymbolicEngineError(
            "Transfer function not available",
            stage="get_transfer_function_expr"
        )

    def get_latex_algebraic_transfer_function(self) -> str | None:
        try:
            if hasattr(self, "algebraic_s_domain_tf"):
                return latex(self.algebraic_s_domain_tf)
            if hasattr(self, "algebraic_tf"):
                return latex(self.algebraic_tf)
        except Exception as e:
            raise SymbolicEngineError(
                f"Failed to generate algebraic TF LaTeX: {e}",
                stage="latex_algebraic_tf"
            )
        return None

# --------------------------------------------------
# BlockDiagramEngine (UNCHANGED)
# --------------------------------------------------
# ↓↓↓  ส่วนนี้คงเดิมตามไฟล์ต้นฉบับของคุณ  ↓↓↓



class BlockDiagramEngine:
    """
    TOPOLOGY-STRICT Block Diagram Engine (Refined Version with Large Signs)
    กฎ:
    1) สร้าง Summing Block (S_i) เฉพาะเมื่อมี Input > 1 เท่านั้น
    2) ขาออกของ S_i มีป้าย x_i และไม่มีเครื่องหมาย
    3) ขาเข้าของ S_i แสดงเฉพาะเครื่องหมาย (+/-) ขนาดใหญ่ (2 เท่า)
    """

    def __init__(self, engine):
        self.engine = engine
        self.equations = engine.equations
        self.x = list(engine.x)
        self.R = engine.R
        self.n = len(self.x)

    def _sanitize_id(self, text):
        import re
        return re.sub(r'[^a-zA-Z0-9_]', '_', str(text))

    def build_block_graph(self):
        from sympy import expand, simplify
        graph = {"nodes": [], "edges": []}
        
        # 1. วิเคราะห์ข้อมูลสมการและ Usage
        usage = {str(xi): 0 for xi in self.x}
        usage["R"] = 0
        eq_data = []

        for i, eq in enumerate(self.equations, start=1):
            xi_sym = self.x[i - 1]
            expr = expand(eq.lhs)
            coeff = expr.coeff(xi_sym)
            if coeff == 0: continue

            rhs = -(expr - coeff * xi_sym) / coeff
            terms = rhs.as_ordered_terms()
            
            inputs = []
            for t in terms:
                if t.has(self.R):
                    usage["R"] += 1
                    inputs.append(("R", simplify(t / self.R)))
                else:
                    for xj in self.x:
                        if t.has(xj):
                            usage[str(xj)] += 1
                            inputs.append((str(xj), simplify(t / xj)))
                            break
            
            needs_sum = len(inputs) > 1
            eq_data.append({
                "i": i, "xi": str(xi_sym), "inputs": inputs, "needs_sum": needs_sum
            })

        # 2. Nodes พื้นฐาน
        graph["nodes"].append({"id": "R", "type": "input", "label": "Input (R)"})
        graph["nodes"].append({"id": "Y", "type": "output", "label": "Output (Y)"})

        # 3. Take-off และ Summing Nodes
        takeoff_nodes = {}
        x_final_source = {}

        for d in eq_data:
            xi = d["xi"]
            if usage[xi] > 1 or xi == str(self.x[-1]):
                tid = f"T_{xi}"
                takeoff_nodes[xi] = tid
                graph["nodes"].append({"id": tid, "type": "takeoff"})
                x_final_source[xi] = tid

            if d["needs_sum"]:
                sid = f"S{d['i']}"
                graph["nodes"].append({"id": sid, "type": "sum"})
                if xi in takeoff_nodes:
                    graph["edges"].append({"from": sid, "to": takeoff_nodes[xi], "label": xi})
                else:
                    x_final_source[xi] = sid

        # 4. Blocks และการเชื่อมสาย (เพิ่มการขยายขนาดเครื่องหมาย)
        for d in eq_data:
            xi = d["xi"]
            needs_sum = d["needs_sum"]
            target_node = f"S{d['i']}" if needs_sum else (takeoff_nodes.get(xi) or "Y")

            for src_name, gain in d["inputs"]:
                gain_str = str(gain)
                if src_name == "R": src_node = "R"
                else:
                    src_node = x_final_source.get(src_name)
                    if not src_node:
                        src_idx = [str(v) for v in self.x].index(src_name) + 1
                        src_node = f"S{src_idx}"

                # --- ส่วนที่ปรับขนาดเครื่องหมาย ---
                raw_sign = "-" if gain_str.startswith("-") else "+"
                # ใช้ HTML-like label เพื่อขยายขนาดเฉพาะเครื่องหมาย (ประมาณ 2 เท่าจากปกติ 12-14pt)
                large_sign = f'<<FONT POINT-SIZE="24">{raw_sign}</FONT>>'
                
                pure_gain = gain_str.lstrip("-")

                # กฎ: ขาเข้า summing block ใส่เฉพาะเครื่องหมายที่ขยายขนาดแล้ว
                input_label = large_sign if target_node.startswith("S") else xi

                if pure_gain == "1":
                    graph["edges"].append({
                        "from": src_node, "to": target_node, "label": input_label
                    })
                else:
                    bid = self._sanitize_id(f"G_{src_name}_to_{xi}")
                    graph["nodes"].append({
                        "id": bid, "type": "block", "label_display": pure_gain
                    })
                    graph["edges"].append({
                        "from": src_node, "to": bid, 
                        "label": src_name if src_node.startswith("S") or src_node == "R" else ""
                    })
                    graph["edges"].append({
                        "from": bid, "to": target_node, "label": input_label
                    })

                if not needs_sum and xi not in takeoff_nodes:
                    x_final_source[xi] = bid if pure_gain != "1" else src_node

            # บังคับขาออกจาก Sum ให้มีป้าย xi
            if needs_sum:
                sid = f"S{d['i']}"
                for e in graph["edges"]:
                    if e["from"] == sid and not e.get("label"):
                        e["label"] = xi

        # 5. Output (Y)
        last_xi = str(self.x[-1])
        if last_xi in x_final_source:
            if not any(e["to"] == "Y" for e in graph["edges"]):
                graph["edges"].append({
                    "from": x_final_source[last_xi], "to": "Y", "label": last_xi
                })

        return graph

    
def block_graph_to_mermaid(graph):
    # เปลี่ยนเป็น LR เพื่อให้อ่านง่ายขึ้นในรูปแบบ Control System
    lines = ["flowchart LR"]

    # ---------- Nodes ----------
    for n in graph["nodes"]:
        nid, t = n["id"], n["type"]
        display_name = n.get("label_display", nid)

        if t == "sum":
            lines.append(f'{nid}(("(Σ)")):::sum')
        elif t == "block":
            lines.append(f'{nid}[" {display_name} "]:::block')
        elif t == "takeoff":
            lines.append(f'{nid}(( )):::takeoff')
        elif t == "input":
            lines.append(f'{nid}(["{display_name}"]):::input')
        elif t == "output":
            lines.append(f'{nid}(["{display_name}"]):::output')

    # ---------- Edges ----------
    for e in graph["edges"]:
        label = e.get("label", "")
        if label:
            lines.append(f'{e["from"]} -- "{label}" --> {e["to"]}')
        else:
            lines.append(f'{e["from"]} --> {e["to"]}')

    # ---------- Styles ----------
    lines += [
        "classDef sum fill:#f9f9f9,stroke:#333,stroke-width:2px;",
        "classDef block fill:#fff,stroke:#333,stroke-width:1px;",
        "classDef input fill:#e1f5fe,stroke:#01579b;",
        "classDef output fill:#fff3e0,stroke:#e65100;",
        "classDef takeoff fill:#333,stroke:#333;",
    ]

    return "\n".join(lines)


