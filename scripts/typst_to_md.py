#!/usr/bin/env python3
"""
Convert Typst spec files to Markdown by parsing both .typ prose and .toml data.

This script:
1. Parses .typ files for prose content (headings, paragraphs, notes)
2. Parses .toml files for structured data (variables, constraints, assumptions)
3. Detects #render_constraint_table() calls to insert tables at correct positions
4. Reads constraint group prefixes from TOML (e.g., "R" -> "CR")
5. Maintains continuous constraint numbering across groups

Usage:
    cd scripts
    source .venv/bin/activate
    python typst_to_md.py                              # Output to spec/
    python typst_to_md.py -o ../others/spec_new_md     # Output to specific dir

Requirements:
    pip install tomli  (or use Python 3.11+ which has tomllib built-in)
"""

import argparse
import re
import sys
from pathlib import Path

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        print("Error: Please install tomli: pip install tomli", file=sys.stderr)
        sys.exit(1)


# =============================================================================
# Expression Rendering (from TOML constraint expressions)
# =============================================================================

def type_to_text(typ) -> str:
    """Convert a type to text."""
    if isinstance(typ, str):
        return typ
    if isinstance(typ, list) and len(typ) == 2:
        return f"{typ[0]}[{typ[1]}]"
    return str(typ)


def expr_to_text(expr, parent_prec: int = 100) -> str:
    """
    Convert a polynomial expression to readable text.

    Expression grammar (from spec/expr.typ):
        <expr> ::= str                           ; variable name
                 | int                           ; constant
                 | ["idx", expr1, expr2]         ; expr1[expr2]
                 | ["not", expr]                 ; 1 - expr
                 | ["+", expr1, expr2, ...]      ; expr1 + expr2 + ...
                 | ["sum", expr1, expr2, expr3]  ; sum from expr1 to expr2 of expr3
                 | ["*", expr1, expr2, ...]      ; expr1 * expr2 * ...
                 | ["/", expr1, expr2]           ; expr1 / expr2
                 | ["^", expr1, expr2]           ; expr1^expr2
                 | ["=", expr1, expr2]           ; expr1 = expr2
                 | [":=", expr1, expr2]          ; expr1 := expr2
                 | ["-", expr]                   ; -expr (unary)
                 | ["-", expr1, expr2, ...]      ; expr1 - expr2 - ... (binary)
                 | ["cast", expr, type]          ; expr::type
    """
    PREC = {
        "idx": 0, "pow": 1, "neg": 2, "cast": 3, "mul": 4,
        "div": 5, "mod": 6, "sum": 7, "not": 8, "add": 9, "sub": 10, "eq": 11,
    }

    def wrap(s: str, prec: int) -> str:
        return f"({s})" if parent_prec < prec else s

    if expr is None or expr == "":
        return ""
    if isinstance(expr, str):
        return expr
    if isinstance(expr, (int, float)):
        return str(expr)

    if isinstance(expr, list) and len(expr) > 0:
        op = expr[0]

        if op == "idx":
            base = expr_to_text(expr[1], PREC["idx"])
            idx = expr_to_text(expr[2], 100)
            return f"{base}[{idx}]"
        elif op == "arr":
            parts = [expr_to_text(e, 100) for e in expr[1:]]
            return "[" + ", ".join(parts) + "]"
        elif op == "opsel":
            return f"⧼{expr[1]}⧽"
        elif op == "mod":
            lhs = expr_to_text(expr[1], PREC["mod"])
            rhs = expr_to_text(expr[2], PREC["mod"])
            return wrap(f"{lhs} mod {rhs}", PREC["mod"])
        elif op == "not":
            inner = expr_to_text(expr[1], PREC["not"])
            return wrap(f"1 - {inner}", PREC["not"])
        elif op == "+":
            parts = [expr_to_text(e, PREC["add"]) for e in expr[1:]]
            return wrap(" + ".join(parts), PREC["add"])
        elif op == "sum":
            var = expr_to_text(expr[1], 100)
            upper = expr_to_text(expr[2], 100)
            body = expr_to_text(expr[3], PREC["sum"])
            return f"Σ_{var}^{upper} {body}"
        elif op == "*":
            parts = [expr_to_text(e, PREC["mul"]) for e in expr[1:]]
            return wrap(" * ".join(parts), PREC["mul"])
        elif op == "/":
            num = expr_to_text(expr[1], PREC["div"])
            den = expr_to_text(expr[2], PREC["div"])
            return wrap(f"{num} / {den}", PREC["div"])
        elif op == "^":
            base = expr_to_text(expr[1], PREC["pow"])
            exp = expr_to_text(expr[2], PREC["pow"])
            return f"{base}^{exp}"
        elif op == "=":
            lhs = expr_to_text(expr[1], PREC["eq"])
            rhs = expr_to_text(expr[2], PREC["eq"])
            return f"{lhs} = {rhs}"
        elif op == ":=":
            lhs = expr_to_text(expr[1], PREC["eq"])
            rhs = expr_to_text(expr[2], PREC["eq"])
            return f"{lhs} := {rhs}"
        elif op == "-":
            if len(expr) == 2:
                inner = expr_to_text(expr[1], PREC["neg"])
                return wrap(f"-{inner}", PREC["neg"])
            else:
                parts = [expr_to_text(e, PREC["sub"]) for e in expr[1:]]
                return wrap(" - ".join(parts), PREC["sub"])
        elif op == "cast":
            inner = expr_to_text(expr[1], PREC["cast"])
            type_str = type_to_text(expr[2])
            return wrap(f"{inner}::{type_str}", PREC["cast"])
        else:
            return str(expr)

    return str(expr)


def iters_to_text(obj: dict) -> str:
    """Extract iterator ranges from a constraint/assumption."""
    iters = []

    if "iter" in obj:
        it = obj["iter"]
        if isinstance(it, list) and len(it) == 3:
            iters.append(f"{it[0]} ∈ [{it[1]}, {it[2]}]")
        elif isinstance(it, list) and len(it) == 2:
            iters.append(f"{it[0]} = {it[1]}")

    if "iters" in obj:
        for it in obj["iters"]:
            if isinstance(it, list) and len(it) == 3:
                iters.append(f"{it[0]} ∈ [{it[1]}, {it[2]}]")
            elif isinstance(it, list) and len(it) == 2:
                iters.append(f"{it[0]} = {it[1]}")

    return ", ".join(iters)


# Chapters in order (from book.typ)
CHAPTERS = [
    ("logup", "LogUp Argument"),
    ("memory", "Memory Argument"),
    ("variables", "Variables"),
    ("signatures", "Signatures"),
    ("is_bit", "IS_BIT Template"),
    ("is_byte", "IS_BYTE Template"),
    ("sign", "SIGN Template"),
    ("add", "ADD/SUB Template"),
    ("neg", "NEG Template"),
    ("decode", "DECODE Table"),
    ("cpu", "CPU Chip"),
    ("cpu32", "CPU32 Chip"),
    ("shift", "SHIFT Chip"),
    ("branch", "BRANCH Chip"),
    ("lt", "LT Chip"),
    ("eq", "EQ Chip"),
    ("mul", "MUL Chip"),
    ("dvrm", "DVRM Chip"),
    ("bitwise", "BITWISE Chips"),
    ("bytewise", "BYTEWISE Chip"),
    ("memw", "MEMW Chip"),
    ("load", "LOAD Chip"),
    ("store", "STORE Chip"),
    ("about_ecalls", "About ECALL"),
    ("halt", "HALT Chip"),
    ("commit", "COMMIT Chip"),
    ("sha256", "SHA256 Accelerator"),
    ("keccak", "KECCAK Accelerator"),
]


def load_toml(path: Path) -> dict:
    """Load a TOML file."""
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        return tomllib.load(f)


def parse_typst_prose(content: str) -> list:
    """
    Parse Typst file and extract prose sections.
    Returns list of (type, content) tuples.
    """
    elements = []

    # Remove multi-line import blocks and top-level #import/#show lines
    content = re.sub(r'^#import[^\n]*\n', '', content, flags=re.MULTILINE)
    content = re.sub(r'^#show:[^\n]*\n', '', content, flags=re.MULTILINE)
    content = re.sub(r'#import[^)]+\)', '', content)

    lines = content.split('\n')
    i = 0
    current_para = []

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Skip empty lines
        if not stripped:
            if current_para:
                elements.append(('para', ' '.join(current_para)))
                current_para = []
            i += 1
            continue

        # Capture render_constraint_table calls to know which group to render
        if stripped.startswith('#render_constraint_table'):
            if current_para:
                elements.append(('para', ' '.join(current_para)))
                current_para = []
            # Extract chip variable (first argument)
            chip_var_match = re.match(r'#render_constraint_table\((\w+)', stripped)
            chip_var = chip_var_match.group(1) if chip_var_match else None
            # Extract group names: handles both single `groups: "g"` and array `groups: ("g1", "g2")`
            groups = []
            array_match = re.search(r'groups:\s*\(([^)]*)\)', stripped)
            if array_match:
                groups = re.findall(r'"([^"]+)"', array_match.group(1))
            else:
                single_match = re.search(r'groups:\s*"([^"]+)"', stripped)
                if single_match:
                    groups = [single_match.group(1)]
            if groups:
                elements.append(('render_constraints', (chip_var, groups)))
            else:
                # No groups specified — render all
                elements.append(('render_constraints', (chip_var, None)))
            i += 1
            continue

        # Capture explicit variable/column table renders
        if stripped.startswith('#render_chip_variable_table') or stripped.startswith('#render_chip_column_table'):
            if current_para:
                elements.append(('para', ' '.join(current_para)))
                current_para = []
            chip_var_match = re.match(r'#render_chip_(?:variable|column)_table\((\w+)', stripped)
            chip_var = chip_var_match.group(1) if chip_var_match else None
            elements.append(('render_variables', chip_var))
            i += 1
            continue

        # Capture explicit assumptions renders
        if stripped.startswith('#render_chip_assumptions'):
            if current_para:
                elements.append(('para', ' '.join(current_para)))
                current_para = []
            chip_var_match = re.match(r'#render_chip_assumptions\((\w+)', stripped)
            chip_var = chip_var_match.group(1) if chip_var_match else None
            elements.append(('render_assumptions', chip_var))
            i += 1
            continue

        # Capture explicit padding table renders
        if stripped.startswith('#render_chip_padding_table'):
            if current_para:
                elements.append(('para', ' '.join(current_para)))
                current_para = []
            chip_var_match = re.match(r'#render_chip_padding_table\((\w+)', stripped)
            chip_var = chip_var_match.group(1) if chip_var_match else None
            elements.append(('render_padding', chip_var))
            i += 1
            continue

        # Skip other function calls (table renders, etc.)
        if stripped.startswith('#render_') or stripped.startswith('#total_'):
            if current_para:
                elements.append(('para', ' '.join(current_para)))
                current_para = []
            i += 1
            continue

        # Skip lines that are just function names (from multi-line imports)
        if re.match(r'^[a-z_]+,?\s*$', stripped) or stripped == ')':
            i += 1
            continue

        # Detect chip loads: #let <varname> = load_chip("src/foo.toml", config)
        load_chip_match = re.match(r'#let\s+(\w+)\s*=\s*load_chip\("([^"]+)"', stripped)
        if load_chip_match:
            if current_para:
                elements.append(('para', ' '.join(current_para)))
                current_para = []
            var_name = load_chip_match.group(1)
            chip_path = load_chip_match.group(2)
            elements.append(('load_chip', (var_name, chip_path)))
            i += 1
            continue

        # Detect chip name aliases: #let <alias> = raw(<chipvar>.name)
        name_alias_match = re.match(r'#let\s+(\w+)\s*=\s*raw\((\w+)\.name\)', stripped)
        if name_alias_match:
            if current_para:
                elements.append(('para', ' '.join(current_para)))
                current_para = []
            alias = name_alias_match.group(1)
            chip_var = name_alias_match.group(2)
            elements.append(('name_alias', (alias, chip_var)))
            i += 1
            continue

        # Skip other Typst commands we don't need
        if stripped.startswith('#') and not stripped.startswith('#rj[') and not stripped.startswith('#et['):
            if current_para:
                elements.append(('para', ' '.join(current_para)))
                current_para = []
            i += 1
            continue

        # Headings (= level 1, == level 2, etc.)
        if stripped.startswith('=') and (len(stripped) == 1 or stripped[len(re.match(r'^=+', stripped).group())] == ' '):
            if current_para:
                elements.append(('para', ' '.join(current_para)))
                current_para = []

            level = len(re.match(r'^=+', stripped).group())
            title = stripped[level:].strip()
            elements.append((f'h{level}', title))
            i += 1
            continue

        # TODO/review notes - extract the content
        todo_match = re.match(r'#(rj|et)\[([^\]]*)\]', stripped)
        if todo_match:
            note_content = todo_match.group(2)
            elements.append(('note', note_content))
            i += 1
            continue

        # Regular text (prose)
        # Clean up inline Typst markup
        text = stripped
        text = re.sub(r'#`([^`]*)`', r'`\1`', text)  # #`code` -> `code`
        text = re.sub(r'@(\w+:\w+:\w+)', r'[\1]', text)  # @ref:to:thing -> [ref:to:thing]
        text = re.sub(r'@(\w+)', r'[\1]', text)  # @ref -> [ref]
        text = re.sub(r'#total_nr_\w+\([^)]+\)', 'N', text)  # #total_nr_xxx(chip) -> N
        text = re.sub(r'#\w+\([^)]*\)', '', text)  # Remove other function calls
        text = re.sub(r'\$([^$]+)\$', r'`\1`', text)  # $math$ -> `math`

        if text and not text.startswith('#'):
            current_para.append(text)

        i += 1

    if current_para:
        elements.append(('para', ' '.join(current_para)))

    return elements


def render_variables_table(chip: dict, config: dict) -> str:
    """Render variables as Markdown tables."""
    variables = chip.get("variables", {})
    if not variables:
        return ""

    lines = []
    category_order = ["input", "output", "auxiliary", "virtual", "multiplicity", "condition"]

    for category in category_order:
        if category not in variables:
            continue

        vars_list = variables[category]
        if not vars_list:
            continue

        lines.append(f"### {category.capitalize()}")
        lines.append("")
        lines.append("| Name | Type | Description |")
        lines.append("|------|------|-------------|")

        for var in vars_list:
            name = f"`{var['name']}`"
            typ = f"`{type_to_text(var.get('type', ''))}`"
            desc = var.get('desc', '').replace('|', '\\|').replace('\n', ' ')
            desc = re.sub(r'#`([^`]*)`', r'`\1`', desc)
            lines.append(f"| {name} | {typ} | {desc} |")

        # Add definitions for virtual variables
        for var in vars_list:
            if "def" in var:
                defn = var["def"]
                lines.append("")
                lines.append(f"**Definition of `{var['name']}`:**")
                if isinstance(defn, dict):
                    if "poly" in defn:
                        lines.append("```")
                        lines.append(f"{var['name']} := {expr_to_text(defn['poly'])}")
                        lines.append("```")
                    elif "polys" in defn:
                        lines.append("```")
                        for p in defn["polys"]:
                            iter_str = ""
                            if "iter" in p:
                                iter_str = f" (when iter={p['iter']})"
                            lines.append(f"{var['name']}{iter_str} := {expr_to_text(p['poly'])}")
                        lines.append("```")
                elif isinstance(defn, (list, str)):
                    lines.append("```")
                    lines.append(f"{var['name']} := {expr_to_text(defn)}")
                    lines.append("```")

        lines.append("")

    return "\n".join(lines)


def render_constraints_table(chip: dict, config: dict, group_filter: str = None, skip_heading: bool = False, start_counter: int = None) -> str:
    """Render constraints as Markdown tables."""
    constraints = chip.get("constraints", {})
    constraint_groups = chip.get("constraint_groups", [])

    if not constraints:
        return ""

    chip_name = chip.get("name", "").upper()
    lines = []
    group_info = {g["name"]: g for g in constraint_groups}

    # Calculate starting counter based on constraints before the filtered group
    if start_counter is not None:
        global_counter = start_counter
    elif group_filter:
        # Count constraints in all groups that come before this one
        global_counter = 1
        for cg in constraint_groups:
            if cg["name"] == group_filter:
                break
            group_constraints = constraints.get(cg["name"], [])
            global_counter += len(group_constraints)
    else:
        global_counter = 1

    for group_name, group_constraints in constraints.items():
        if group_filter and group_name != group_filter:
            continue
        if not group_constraints:
            continue

        group = group_info.get(group_name, {"name": group_name})
        # Get prefix from TOML constraint_groups (e.g., "R" -> "CR", "M" -> "CM")
        # The base is always "C" for Constraint, plus the group's prefix if defined
        group_prefix = "C" + group.get("prefix", "")

        # Check if any constraint has multiplicity or polynomial
        has_mult = any("multiplicity" in c for c in group_constraints)
        has_iter = any(iters_to_text(c) for c in group_constraints)
        has_poly = any(c.get("kind") == "arith" and ("poly" in c or "polys" in c) for c in group_constraints)

        if not skip_heading:
            lines.append(f"### {group_name}")
            lines.append("")

        # Build header based on columns needed
        if has_iter and has_mult:
            lines.append("| Tag | Range | Description | Multiplicity |")
            lines.append("|-----|-------|-------------|--------------|")
        elif has_iter:
            lines.append("| Tag | Range | Description |")
            lines.append("|-----|-------|-------------|")
        elif has_mult:
            lines.append("| Tag | Description | Multiplicity |")
            lines.append("|-----|-------------|--------------|")
        else:
            lines.append("| Tag | Description |")
            lines.append("|-----|-------------|")

        for i, constraint in enumerate(group_constraints, 1):
            # Always auto-generate ref with chip and group prefix (like shiroa does)
            iters = iters_to_text(constraint)
            iter_suffix = ".i" if iters else ""

            ref = f"{chip_name}-{group_prefix}{global_counter}{iter_suffix}" if chip_name else f"{group_prefix}{global_counter}{iter_suffix}"

            kind = constraint.get("kind", "")
            tag = constraint.get("tag", "")

            # Build description based on kind
            cond = constraint.get("cond")
            cond_str = f"{expr_to_text(cond)} ⇒ " if cond else ""

            if kind == "interaction":
                inputs = ", ".join(expr_to_text(inp) for inp in constraint.get("input", []))
                output = constraint.get("output")
                if output:
                    desc = f"{cond_str}`{tag}[{expr_to_text(output)}; {inputs}]`"
                else:
                    desc = f"{cond_str}`{tag}[{inputs}]`"
            elif kind == "arith":
                desc = constraint.get("constraint", "")
                desc = desc.replace("$", "").replace("#", "")
                if cond_str:
                    desc = f"{cond_str}{desc}"
            elif kind == "template":
                inputs = ", ".join(expr_to_text(inp) for inp in constraint.get("input", []))
                output = constraint.get("output")
                if output:
                    desc = f"{cond_str}`{tag}<{expr_to_text(output)}; {inputs}>`"
                else:
                    desc = f"{cond_str}`{tag}<{inputs}>`"
            else:
                desc = str(constraint)

            # Get range and multiplicity
            mult = expr_to_text(constraint.get("multiplicity", ""))

            # Build row based on columns
            if has_iter and has_mult:
                lines.append(f"| `{ref}` | {iters} | {desc} | {mult} |")
            elif has_iter:
                lines.append(f"| `{ref}` | {iters} | {desc} |")
            elif has_mult:
                lines.append(f"| `{ref}` | {desc} | {mult} |")
            else:
                lines.append(f"| `{ref}` | {desc} |")

            # Add polynomial constraint if present (for arith constraints)
            if kind == "arith" and ("poly" in constraint or "polys" in constraint):
                if "poly" in constraint:
                    poly_str = expr_to_text(constraint["poly"])
                    if has_iter and has_mult:
                        lines.append(f"| | | _polynomial:_ `{poly_str} = 0` | |")
                    elif has_iter:
                        lines.append(f"| | | _polynomial:_ `{poly_str} = 0` |")
                    elif has_mult:
                        lines.append(f"| | _polynomial:_ `{poly_str} = 0` | |")
                    else:
                        lines.append(f"| | _polynomial:_ `{poly_str} = 0` |")
                elif "polys" in constraint:
                    for poly in constraint["polys"]:
                        poly_str = expr_to_text(poly)
                        if has_iter and has_mult:
                            lines.append(f"| | | _polynomial:_ `{poly_str} = 0` | |")
                        elif has_iter:
                            lines.append(f"| | | _polynomial:_ `{poly_str} = 0` |")
                        elif has_mult:
                            lines.append(f"| | _polynomial:_ `{poly_str} = 0` | |")
                        else:
                            lines.append(f"| | _polynomial:_ `{poly_str} = 0` |")

            global_counter += 1

        lines.append("")

    return "\n".join(lines)


def render_assumptions_table(chip: dict, config: dict) -> str:
    """Render assumptions as Markdown table."""
    assumptions = chip.get("assumptions", [])
    if not assumptions:
        return ""

    chip_name = chip.get("name", "").upper()
    prefix = f"{chip_name}-A" if chip_name else "A"

    lines = []
    lines.append("| Tag | Range | Description |")
    lines.append("|-----|-------|-------------|")

    for i, assumption in enumerate(assumptions, 1):
        iters = iters_to_text(assumption)
        iter_suffix = ".i" if iters else ""
        ref = f"{chip_name}-A{i}{iter_suffix}" if chip_name else f"A{i}{iter_suffix}"
        desc = assumption.get("desc", "").replace("|", "\\|")
        lines.append(f"| `{ref}` | {iters} | {desc} |")

    lines.append("")
    return "\n".join(lines)


def render_padding_table(chip: dict, config: dict) -> str:
    """Render padding data as Markdown table.

    Padding values live on each variable as a `pad` attribute (mirrors
    `render_chip_padding_table` in spec/chip.typ): instantiated,
    non-preprocessed variables only.
    """
    var_cfg = config.get("variables", {})
    instantiated = var_cfg.get("categories", {}).get("instantiated", [])
    preprocessed_labels = {
        t["label"] for t in var_cfg.get("types", []) if t.get("preprocessed", False)
    }

    rows = []
    for category in instantiated:
        for var in chip.get("variables", {}).get(category, []):
            var_type = var.get("type")
            if isinstance(var_type, str) and var_type in preprocessed_labels:
                continue
            if "pad" in var:
                rows.append((var["name"], expr_to_text(var["pad"])))

    # Legacy schema fallback: top-level `padding` table.
    for col_name, value in chip.get("padding", {}).items():
        rows.append((col_name, str(value)))

    if not rows:
        return ""

    lines = []
    lines.append("| Column | Padding value |")
    lines.append("|--------|---------------|")
    for name, value in rows:
        lines.append(f"| `{name}` | `{value}` |")

    lines.append("")
    return "\n".join(lines)


def convert_chapter(typ_path: Path, toml_path: Path, title: str, config: dict, spec_dir: Path = None) -> str:
    """Convert a chapter from .typ and .toml to Markdown."""
    lines = [f"# {title}", ""]

    # Load default TOML data (may be empty for prose-only or multi-chip files)
    default_chip = load_toml(toml_path)

    # Chip registry: variable_name -> chip_data
    chips = {}
    if default_chip:
        chips['chip'] = default_chip

    # Name alias registry: alias -> chip_var_name (from #let alias = raw(chipvar.name))
    name_aliases = {}

    def reset_chip_state():
        return {
            'rendered_columns': False,
            'rendered_assumptions': False,
            'rendered_constraints': False,
            'rendered_constraint_groups': set(),
            'constraint_counter': 1,
        }

    # State registry: variable_name -> render state
    states = {}
    if default_chip:
        states['chip'] = reset_chip_state()

    def resolve_chip(var_name):
        """Resolve chip variable name to (chip_data, state)."""
        if var_name and var_name in chips:
            if var_name not in states:
                states[var_name] = reset_chip_state()
            return chips[var_name], states[var_name]
        # Fallback to default 'chip' key
        if 'chip' in chips:
            if 'chip' not in states:
                states['chip'] = reset_chip_state()
            return chips['chip'], states['chip']
        # Fallback to first loaded chip
        if chips:
            first_key = next(iter(chips))
            if first_key not in states:
                states[first_key] = reset_chip_state()
            return chips[first_key], states[first_key]
        return {}, reset_chip_state()

    # Parse Typst prose
    if typ_path.exists():
        typst_content = typ_path.read_text()
        elements = parse_typst_prose(typst_content)

        for elem_type, content in elements:
            if elem_type == 'load_chip':
                var_name, chip_path = content
                chip_toml_path = spec_dir / chip_path if spec_dir else Path(chip_path)
                chips[var_name] = load_toml(chip_toml_path)
                states[var_name] = reset_chip_state()
                continue

            if elem_type == 'name_alias':
                alias, chip_var = content
                name_aliases[alias] = chip_var
                continue

            if elem_type.startswith('h'):
                level = int(elem_type[1])
                lines.append("")
                heading_text = content
                # Replace Typst variable references (#varname) with chip names
                for alias, chip_var in name_aliases.items():
                    if f'#{alias}' in heading_text and chip_var in chips:
                        chip_name = chips[chip_var].get('name', alias)
                        heading_text = heading_text.replace(f'#{alias}', f'`{chip_name}`')
                # Offset by +1 since the chapter title already uses #
                lines.append("#" * (level + 1) + " " + heading_text)
                lines.append("")

            elif elem_type == 'render_variables':
                chip_var = content
                chip_data, st = resolve_chip(chip_var)
                if chip_data and not st['rendered_columns']:
                    lines.append(render_variables_table(chip_data, config))
                    st['rendered_columns'] = True

            elif elem_type == 'render_assumptions':
                chip_var = content
                chip_data, st = resolve_chip(chip_var)
                if chip_data and not st['rendered_assumptions']:
                    lines.append(render_assumptions_table(chip_data, config))
                    st['rendered_assumptions'] = True

            elif elem_type == 'render_padding':
                chip_var = content
                chip_data, st = resolve_chip(chip_var)
                if chip_data:
                    padding = render_padding_table(chip_data, config)
                    if padding.strip():
                        lines.append(padding)

            elif elem_type == 'render_constraints':
                chip_var, group_names = content
                chip_data, st = resolve_chip(chip_var)
                if chip_data:
                    if group_names is None:
                        # Render all groups
                        group_names = [cg["name"] for cg in chip_data.get("constraint_groups", [])]
                    for group_name in group_names:
                        if group_name not in st['rendered_constraint_groups']:
                            group_table = render_constraints_table(
                                chip_data, config,
                                group_filter=group_name,
                                skip_heading=True,
                                start_counter=st['constraint_counter'],
                            )
                            if group_table.strip():
                                lines.append(group_table)
                            st['rendered_constraint_groups'].add(group_name)
                        st['constraint_counter'] += len(
                            chip_data.get("constraints", {}).get(group_name, [])
                        )

            elif elem_type == 'para':
                lines.append(content)
                lines.append("")

            elif elem_type == 'note':
                lines.append(f"> **Note:** {content}")
                lines.append("")

    # Fallback: render any TOML data not yet triggered by explicit render calls
    for var_name, chip_data in chips.items():
        if var_name not in states:
            states[var_name] = reset_chip_state()
        st = states[var_name]

        if chip_data.get("variables") and not st['rendered_columns']:
            lines.append("## Columns")
            lines.append("")
            lines.append(render_variables_table(chip_data, config))

        if chip_data.get("assumptions") and not st['rendered_assumptions']:
            lines.append("## Assumptions")
            lines.append("")
            lines.append(render_assumptions_table(chip_data, config))

        if chip_data.get("constraints"):
            all_groups_ordered = [cg["name"] for cg in chip_data.get("constraint_groups", [])]
            remaining_groups = [g for g in all_groups_ordered if g not in st['rendered_constraint_groups']]

            if remaining_groups and not st['rendered_constraints']:
                lines.append("## Constraints")
                lines.append("")

            for group_name in remaining_groups:
                group_table = render_constraints_table(
                    chip_data, config,
                    group_filter=group_name,
                    start_counter=st['constraint_counter'],
                )
                if group_table.strip():
                    lines.append(group_table)
                st['constraint_counter'] += len(
                    chip_data.get("constraints", {}).get(group_name, [])
                )

    result = "\n".join(lines)
    result = re.sub(r'\n{3,}', '\n\n', result)
    # Clean up remaining Typst artifacts
    result = re.sub(r'#\w+\[[^\]]*\]', '', result)  # #rj[...], #et[...]
    result = re.sub(r'#\w+', '', result)  # #nr_variables etc
    return result.strip()


def main():
    parser = argparse.ArgumentParser(
        description="Convert Typst spec to Markdown",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--spec-dir", "-s",
        type=Path,
        default=None,
        help="Path to spec directory (default: ../spec)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=None,
        help="Output directory (default: spec directory)"
    )

    args = parser.parse_args()

    script_dir = Path(__file__).parent

    spec_dir = args.spec_dir
    if spec_dir is None:
        spec_dir = script_dir / "../spec"
    spec_dir = spec_dir.resolve()

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = script_dir / "../docs/spec"
    output_dir = output_dir.resolve()

    if not spec_dir.exists():
        print(f"ERROR: Spec directory not found: {spec_dir}", file=sys.stderr)
        return 1

    # Load config
    config_path = spec_dir / "src" / "config.toml"
    config = load_toml(config_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading from: {spec_dir}")
    print(f"Writing to: {output_dir}")
    print()

    all_content = []

    for name, title in CHAPTERS:
        typ_path = spec_dir / f"{name}.typ"
        toml_path = spec_dir / "src" / f"{name}.toml"

        print(f"Converting: {name} ({title})")

        try:
            markdown = convert_chapter(typ_path, toml_path, title, config, spec_dir=spec_dir)

            output_file = output_dir / f"{name}.md"
            output_file.write_text(markdown)

            all_content.append(markdown)

        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()

    # Combined file
    combined_file = output_dir / "spec_full.md"
    combined = "# Lambda VM Specification\n\n"
    combined += "\n\n---\n\n".join(all_content)
    combined_file.write_text(combined)
    print(f"\nCombined: {combined_file}")

    print(f"\nDone! Converted {len(all_content)} chapters.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
