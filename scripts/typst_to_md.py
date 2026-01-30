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
        "div": 5, "sum": 6, "not": 7, "add": 8, "sub": 9, "eq": 10,
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
    ("memory", "Memory Argument"),
    ("variables", "Variables"),
    ("is_bit", "IS_BIT Template"),
    ("add", "ADD/SUB Template"),
    ("decode", "DECODE Table"),
    ("cpu", "CPU Chip"),
    ("shift", "SHIFT Chip"),
    ("branch", "BRANCH Chip"),
    ("memw", "MEMW Chip"),
    ("lt", "LT Chip"),
    ("mul", "MUL Chip"),
    ("dvrm", "DVRM Chip"),
    ("load", "LOAD Chip"),
    ("ecall", "ECALL Chips"),
    ("bitwise", "BITWISE Chips"),
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

    # Remove imports and let bindings at the start
    content = re.sub(r'^#import[^\n]*\n', '', content, flags=re.MULTILINE)
    content = re.sub(r'^#let[^\n]*\n', '', content, flags=re.MULTILINE)
    content = re.sub(r'^#show:[^\n]*\n', '', content, flags=re.MULTILINE)

    # Remove multi-line import blocks
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
            # Extract group name: #render_constraint_table(chip, config, groups: "range")
            match = re.search(r'groups:\s*"([^"]+)"', stripped)
            if match:
                elements.append(('render_constraints', match.group(1)))
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

        # Skip other Typst commands we don't need
        if stripped.startswith('#') and not stripped.startswith('#rj[') and not stripped.startswith('#et['):
            if current_para:
                elements.append(('para', ' '.join(current_para)))
                current_para = []
            i += 1
            continue

        # Headings
        if stripped.startswith('=='):
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


def convert_chapter(typ_path: Path, toml_path: Path, title: str, config: dict) -> str:
    """Convert a chapter from .typ and .toml to Markdown."""
    lines = [f"# {title}", ""]

    # Load TOML data
    chip = load_toml(toml_path)

    # Track what sections we've rendered from TOML
    rendered_columns = False
    rendered_assumptions = False
    rendered_constraints = False
    rendered_constraint_groups = set()

    # Parse Typst prose
    if typ_path.exists():
        typst_content = typ_path.read_text()
        elements = parse_typst_prose(typst_content)

        for elem_type, content in elements:
            if elem_type.startswith('h'):
                level = int(elem_type[1])
                lines.append("")
                lines.append("#" * level + " " + content)
                lines.append("")

                # Render TOML data after relevant headings
                content_lower = content.lower()
                if 'column' in content_lower and chip and not rendered_columns:
                    lines.append(render_variables_table(chip, config))
                    rendered_columns = True
                elif 'assumption' in content_lower and chip and not rendered_assumptions:
                    lines.append(render_assumptions_table(chip, config))
                    rendered_assumptions = True
                elif content_lower == "constraints" and chip:
                    # Mark that we've hit the Constraints section
                    rendered_constraints = True

            elif elem_type == 'render_constraints' and chip:
                # Render the constraint group specified in the typst file
                group_name = content
                if group_name not in rendered_constraint_groups:
                    # Skip heading since prose already has the section title
                    group_table = render_constraints_table(chip, config, group_filter=group_name, skip_heading=True)
                    if group_table.strip():
                        lines.append(group_table)
                        rendered_constraint_groups.add(group_name)

            elif elem_type == 'para':
                lines.append(content)
                lines.append("")

            elif elem_type == 'note':
                lines.append(f"> **Note:** {content}")
                lines.append("")

    # Render any TOML data that wasn't triggered by prose headings
    if chip:
        if chip.get("variables") and not rendered_columns:
            lines.append("## Columns")
            lines.append("")
            lines.append(render_variables_table(chip, config))

        if chip.get("assumptions") and not rendered_assumptions:
            lines.append("## Assumptions")
            lines.append("")
            lines.append(render_assumptions_table(chip, config))

        if chip.get("constraints"):
            # Get all constraint groups from TOML
            all_groups = set(chip.get("constraints", {}).keys())
            remaining_groups = all_groups - rendered_constraint_groups

            if remaining_groups and not rendered_constraints:
                # No prose Constraints section existed, add one
                lines.append("## Constraints")
                lines.append("")

            # Render any constraint groups not already rendered inline
            for group_name in remaining_groups:
                group_table = render_constraints_table(chip, config, group_filter=group_name)
                if group_table.strip():
                    lines.append(group_table)

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
        output_dir = spec_dir
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
            markdown = convert_chapter(typ_path, toml_path, title, config)

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
