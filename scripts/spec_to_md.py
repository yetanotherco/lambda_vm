#!/usr/bin/env python3
"""
Convert Typst spec TOML files to Markdown.

Usage:
    # First, extract spec files from the spec/main branch:
    git show origin/spec/main:spec/src/config.toml > /tmp/spec/config.toml
    git show origin/spec/main:spec/src/cpu.toml > /tmp/spec/cpu.toml
    # etc.

    # Then run:
    python scripts/spec_to_md.py /tmp/spec/config.toml /tmp/spec/cpu.toml

    # Or convert all chips:
    python scripts/spec_to_md.py /tmp/spec/config.toml /tmp/spec/*.toml

    # Output to a specific directory:
    python scripts/spec_to_md.py --output-dir docs/spec /tmp/spec/config.toml /tmp/spec/*.toml
"""

import argparse
import sys
from pathlib import Path

# Python 3.11+ has tomllib in stdlib, fallback to tomli for older versions
try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        print("Error: Please install tomli: pip install tomli", file=sys.stderr)
        sys.exit(1)


# =============================================================================
# Expression Rendering
# =============================================================================

def expr_to_text(expr: any, parent_prec: int = 100) -> str:
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
        "idx": 0,
        "pow": 1,
        "neg": 2,
        "cast": 3,
        "mul": 4,
        "div": 5,
        "sum": 6,
        "not": 7,
        "add": 8,
        "sub": 9,
        "eq": 10,
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
            # expr1[expr2]
            base = expr_to_text(expr[1], PREC["idx"])
            idx = expr_to_text(expr[2], 100)
            return f"{base}[{idx}]"

        elif op == "not":
            # 1 - expr
            inner = expr_to_text(expr[1], PREC["not"])
            return wrap(f"1 - {inner}", PREC["not"])

        elif op == "+":
            # expr1 + expr2 + ...
            parts = [expr_to_text(e, PREC["add"]) for e in expr[1:]]
            return wrap(" + ".join(parts), PREC["add"])

        elif op == "sum":
            # Σ from expr1 to expr2 of expr3
            var = expr_to_text(expr[1], 100)
            upper = expr_to_text(expr[2], 100)
            body = expr_to_text(expr[3], PREC["sum"])
            return f"Σ_{var}^{upper} {body}"

        elif op == "*":
            # expr1 * expr2 * ...
            parts = [expr_to_text(e, PREC["mul"]) for e in expr[1:]]
            return wrap(" * ".join(parts), PREC["mul"])

        elif op == "/":
            # expr1 / expr2
            num = expr_to_text(expr[1], PREC["div"])
            den = expr_to_text(expr[2], PREC["div"])
            return wrap(f"{num} / {den}", PREC["div"])

        elif op == "^":
            # expr1^expr2
            base = expr_to_text(expr[1], PREC["pow"])
            exp = expr_to_text(expr[2], PREC["pow"])
            return f"{base}^{exp}"

        elif op == "=":
            # expr1 = expr2
            lhs = expr_to_text(expr[1], PREC["eq"])
            rhs = expr_to_text(expr[2], PREC["eq"])
            return f"{lhs} = {rhs}"

        elif op == ":=":
            # expr1 := expr2
            lhs = expr_to_text(expr[1], PREC["eq"])
            rhs = expr_to_text(expr[2], PREC["eq"])
            return f"{lhs} := {rhs}"

        elif op == "-":
            if len(expr) == 2:
                # Unary negation
                inner = expr_to_text(expr[1], PREC["neg"])
                return wrap(f"-{inner}", PREC["neg"])
            else:
                # Binary subtraction
                parts = [expr_to_text(e, PREC["sub"]) for e in expr[1:]]
                return wrap(" - ".join(parts), PREC["sub"])

        elif op == "cast":
            # expr::type
            inner = expr_to_text(expr[1], PREC["cast"])
            type_str = type_to_text(expr[2])
            return wrap(f"{inner}::{type_str}", PREC["cast"])

        else:
            # Unknown operator, render as-is
            return str(expr)

    return str(expr)


def type_to_text(typ: any) -> str:
    """Convert a type to text."""
    if isinstance(typ, str):
        return typ
    if isinstance(typ, list) and len(typ) == 2:
        return f"{typ[0]}[{typ[1]}]"
    return str(typ)


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


# =============================================================================
# Markdown Generation
# =============================================================================

def escape_md(s: str) -> str:
    """Escape pipe characters for Markdown tables."""
    if s is None:
        return ""
    return str(s).replace("|", "\\|").replace("\n", " ")


def render_variables_table(variables: dict, config: dict) -> str:
    """Render variables as Markdown tables, grouped by category."""
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
            desc = escape_md(var.get('desc', ''))
            # Clean up Typst markup in descriptions
            desc = desc.replace('#`', '`').replace('`#', '`')
            lines.append(f"| {name} | {typ} | {desc} |")

        # Add definition if present (for virtual variables)
        for var in vars_list:
            if "def" in var:
                defn = var["def"]
                lines.append("")
                lines.append(f"**Definition of `{var['name']}`:**")
                if isinstance(defn, dict):
                    if "poly" in defn:
                        lines.append(f"```")
                        lines.append(f"{var['name']} := {expr_to_text(defn['poly'])}")
                        lines.append(f"```")
                    elif "polys" in defn:
                        lines.append(f"```")
                        for i, p in enumerate(defn["polys"]):
                            iter_str = ""
                            if "iter" in p:
                                iter_str = f" (when iter={p['iter']})"
                            lines.append(f"{var['name']}{iter_str} := {expr_to_text(p['poly'])}")
                        lines.append(f"```")
                elif isinstance(defn, (list, str)):
                    lines.append(f"```")
                    lines.append(f"{var['name']} := {expr_to_text(defn)}")
                    lines.append(f"```")

        lines.append("")

    return "\n".join(lines)


def render_assumptions_table(assumptions: list) -> str:
    """Render assumptions as a Markdown table."""
    if not assumptions:
        return ""

    lines = []
    lines.append("## Assumptions")
    lines.append("")
    lines.append("| Ref | Range | Description |")
    lines.append("|-----|-------|-------------|")

    for i, assumption in enumerate(assumptions, 1):
        ref = assumption.get("ref", f"A{i}")
        iters = iters_to_text(assumption)
        desc = escape_md(assumption.get("desc", ""))
        lines.append(f"| `{ref}` | {iters} | {desc} |")

    lines.append("")
    return "\n".join(lines)


def render_constraints_table(constraints: dict, constraint_groups: list) -> str:
    """Render constraints as Markdown tables, grouped by constraint group."""
    if not constraints:
        return ""

    lines = []
    lines.append("## Constraints")
    lines.append("")

    # Build group lookup
    group_info = {g["name"]: g for g in constraint_groups}

    for group_name, group_constraints in constraints.items():
        if not group_constraints:
            continue

        group = group_info.get(group_name, {"name": group_name})
        prefix = group.get("prefix", "")
        group_desc = group.get("desc", "")

        lines.append(f"### {group_name}")
        if group_desc:
            lines.append(f"_{group_desc}_")
        lines.append("")

        # Determine columns needed
        has_multiplicity = any("multiplicity" in c for c in group_constraints)
        has_iter = any(iters_to_text(c) for c in group_constraints)

        # Build header
        if has_iter and has_multiplicity:
            header = "| Ref | Kind | Range | Description | Multiplicity |"
            separator = "|-----|------|-------|-------------|--------------|"
        elif has_iter:
            header = "| Ref | Kind | Range | Description |"
            separator = "|-----|------|-------|-------------|"
        elif has_multiplicity:
            header = "| Ref | Kind | Description | Multiplicity |"
            separator = "|-----|------|-------------|--------------|"
        else:
            header = "| Ref | Kind | Description |"
            separator = "|-----|------|-------------|"

        lines.append(header)
        lines.append(separator)

        for i, constraint in enumerate(group_constraints, 1):
            ref = constraint.get("ref", f"{prefix}{i}")
            kind = constraint.get("kind", "")
            tag = constraint.get("tag", "")
            iters = iters_to_text(constraint)
            mult = expr_to_text(constraint.get("multiplicity", ""))

            # Build description based on kind
            if kind == "interaction":
                inputs = ", ".join(expr_to_text(inp) for inp in constraint.get("input", []))
                output = constraint.get("output")
                if output:
                    desc = f"`{tag}[{expr_to_text(output)}; {inputs}]`"
                else:
                    desc = f"`{tag}[{inputs}]`"

            elif kind == "arith":
                desc = escape_md(constraint.get("constraint", ""))
                # Clean up Typst math markup
                desc = desc.replace("$", "").replace("#", "")

            elif kind == "template":
                inputs = ", ".join(expr_to_text(inp) for inp in constraint.get("input", []))
                output = constraint.get("output")
                cond = constraint.get("cond")
                cond_str = f"{expr_to_text(cond)} ⇒ " if cond else ""
                if output:
                    desc = f"{cond_str}`{tag}<{expr_to_text(output)}; {inputs}>`"
                else:
                    desc = f"{cond_str}`{tag}<{inputs}>`"

            else:
                desc = str(constraint)

            # Build row
            row = f"| `{ref}` | {kind} |"
            if has_iter:
                row += f" {iters} |"
            row += f" {desc} |"
            if has_multiplicity:
                row += f" {mult} |"

            lines.append(row)

            # Add polynomial constraint if present
            if kind == "arith" and ("poly" in constraint or "polys" in constraint):
                if "poly" in constraint:
                    poly_str = expr_to_text(constraint["poly"])
                    lines.append(f"| | | _polynomial:_ `{poly_str} = 0` |" + (" |" if has_multiplicity else ""))
                elif "polys" in constraint:
                    for poly in constraint["polys"]:
                        poly_str = expr_to_text(poly)
                        lines.append(f"| | | _polynomial:_ `{poly_str} = 0` |" + (" |" if has_multiplicity else ""))

            # Add description if present
            if "desc" in constraint and kind == "arith":
                desc_text = escape_md(constraint["desc"])
                lines.append(f"| | | _note:_ {desc_text} |" + (" |" if has_multiplicity else ""))

        lines.append("")

    return "\n".join(lines)


def chip_to_markdown(chip: dict, config: dict) -> str:
    """Convert a chip TOML to Markdown."""
    lines = []

    name = chip.get("name", "Unknown")
    lines.append(f"# {name} Chip")
    lines.append("")

    # Variables
    variables = chip.get("variables", {})
    if variables:
        lines.append("## Columns")
        lines.append("")
        lines.append(render_variables_table(variables, config))

    # Assumptions
    assumptions = chip.get("assumptions", [])
    if assumptions:
        lines.append(render_assumptions_table(assumptions))

    # Constraints
    constraints = chip.get("constraints", {})
    constraint_groups = chip.get("constraint_groups", [])
    if constraints:
        lines.append(render_constraints_table(constraints, constraint_groups))

    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================

def load_toml(path: Path) -> dict:
    """Load a TOML file."""
    with open(path, "rb") as f:
        return tomllib.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Convert Typst spec TOML files to Markdown",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to config.toml"
    )
    parser.add_argument(
        "chips",
        type=Path,
        nargs="+",
        help="Paths to chip TOML files (e.g., cpu.toml, lt.toml)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=None,
        help="Output directory for Markdown files (default: stdout)"
    )

    args = parser.parse_args()

    # Load config
    config = load_toml(args.config)

    # Process each chip
    for chip_path in args.chips:
        # Skip config.toml if passed as chip
        if chip_path.name == "config.toml":
            continue

        # Skip non-chip TOML files
        if chip_path.name in ("page.toml", "theme-style.toml"):
            continue

        try:
            chip = load_toml(chip_path)
        except Exception as e:
            print(f"Warning: Failed to load {chip_path}: {e}", file=sys.stderr)
            continue

        # Check if it's a valid chip file (has 'name' field)
        if "name" not in chip:
            continue

        md_content = chip_to_markdown(chip, config)

        if args.output_dir:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            output_path = args.output_dir / f"{chip_path.stem}.md"
            with open(output_path, "w") as f:
                f.write(md_content)
            print(f"Generated: {output_path}")
        else:
            print(md_content)
            print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
