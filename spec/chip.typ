#import "expr.typ": expr_to_code, expr_to_math, type_to_code

/// Computes the total number of variables in a `chip`
#let total_nr_variables(chip) = {
  return chip.variables.values().flatten().len()
}

// Computes the total number of columns instantiated by `chip`
#let total_nr_instantiated_columns(chip, config) = {
  return chip
  .variables
  .pairs()
  .filter(pair => pair.at(0) in config.variables.categories.instantiated)
  .map(pair => pair.at(1))
  .flatten()
  .map(var => {
    let (label, factor) = if type(var.type) == array {
      (var.type.at(0), var.type.at(1))
    } else {
      (var.type, 1)
    }
    config.variables.types.filter(type => type.label == label).first().subtypes.len() * factor
  })
  .sum()
}

/// Generates a table listing `chip`'s columns.
#let render_chip_column_table(chip, config) = {
  // Group variables by category
  show figure: set block(breakable: true)
  figure(table(
    columns: (auto, auto, 1fr),
    inset: 6pt,
    align: left + top,
    stroke: none,
    table.header([*Label*], [*Type*], [*Description*]),
    table.hline(stroke: stroke(thickness: 2pt)),
    ..for (cat, vars) in chip.variables.pairs() {
      ([#emph(cat)], [], [], table.hline(stroke: .6pt))
      for var in vars {
        ([#raw(var.name)], [#type_to_code(var.type)], [#eval(var.desc, mode: "markup")])
        for (i, poly) in var.at("polys", default: ()).enumerate() {
          (if i == 0 { emph[def] }, [], expr_to_math(("=", ("idx", var.name, i), poly)))
        }
        if "poly" in var {
          (emph[def], [], expr_to_math(var.poly))
        }
      }
      ([], [], [])
    },
  ), caption: [Column overview of #chip.name chip.])
}

#let cref(obj, body) = {
  if "ref" in obj {
    [#body#label(obj.ref)]
  } else {
    body
  }
}

// Render a range if `obj` contains one.
#let interval(obj) = {
  if "range" in obj {
    [#raw(obj.range.at(0)) #sym.in` [`#obj.range.at(1)`,`#obj.range.at(2)`]`]
  } else { return [] }
}

#let args_interaction_like(input, output) = {
  if output != none {
    expr_to_code(output) + `; `
  } else {
    ``
  } + input.map(expr_to_code).join(`, `)
}

#let render_chip_assumptions(chip, config) = {
  let tag(assumption) = {
    let index = if "range" in assumption { "." + assumption.range.at(0) } else { "" }
    let lbl = [#chip.name\-A]
    show figure: (it) => align(left, block[#lbl#context it.counter.display()#index])
    cref(assumption)[#figure(kind: "assumption", numbering: (i) => [#lbl#i#index], supplement: [], [])]
  }

  figure(table(
    columns: (auto, auto, 1fr),
    inset: 6pt,
    align: (top + left, top + left, top + left),
    stroke: none,
    table.header([*Tag*], [*Range*], [*Description*]),
    table.hline(stroke: stroke(thickness: 2pt)),
    ..for assumption in chip.assumptions {
      ([#tag(assumption)], [#interval(assumption)], [#eval(assumption.desc, mode: "markup")])
    },
  ), caption: [Assumption overview of #chip.name chip.])
}

/// Generates a table listing all interactions initiated by `chip`'s.
#let render_constraint_table(chip, config, groups: none) = {
  let all_groups = chip.constraint_groups.map(group => group.name);
  if groups == none {
    // render all
    groups = all_groups
  } else if type(groups) == str {
    groups = (groups,)
  }
  assert(groups.all(group => group in all_groups), message: "unknown group")

  // Find the group definition in the constraint_groups
  let lookup_group(name) = chip.constraint_groups.filter((g) => g.name == name).at(0, default: (name: name))

  /// Render the contraint's tag.
  let tag(constraint, group) = {
    let index = if "range" in constraint { "." + constraint.range.at(0) } else { "" }
    let prefix = if "prefix" in group { group.prefix }
    let lbl = [#chip.name\-C#prefix]
    show figure: (it) => align(left, block[#lbl#context it.counter.display()#index])
    cref(constraint)[#figure(kind: "constraint", numbering: (i) => [#lbl#i#index], supplement: [], [])]
  }

  /// Generates a representation of `constraint`
  let repr_constraint(constraint) = {
    let kind = constraint.kind

    if kind == "interaction" {
      raw(constraint.tag) + `[` + args_interaction_like(constraint.input, constraint.at("output", default: none)) + `]`
    } else if kind == "arith" {
      [#eval(constraint.constraint, mode: "markup")]
    } else if kind == "template" {
      let cond = if "cond" in constraint {
        $#expr_to_math(constraint.cond) arrow.r.double$ + " "
      }
      cond + raw(constraint.tag) + `<` + args_interaction_like(constraint.input, constraint.at("output", default: none)) + `>`
    } else {
      assert(false, message: "illegal constraint format: " + kind)
    }
  }

  // Whether constraint has polynomial constraints
  let has_polynomial_constraints(constraint) = {
    constraint.kind == "arith" and ("poly" in constraint or "polys" in constraint)
  }

  // Whether constraint has a "desc" field we need to render separately
  let has_extra_description(constraint) = {
    constraint.kind == "arith" and "desc" in constraint
  }

  // Rendering polynomial constraints
  let render_polynomial_constraints(constraint) = {
    assert(constraint.kind == "arith", message: "Only arith needs extra rows")
    let polys = if "poly" in constraint {
      (constraint.poly,)
    } else {
      constraint.polys
    }

    (..for poly in polys {
      ([_polynomial constraint_], [], $#expr_to_math(poly) = 0$, [])
    },)
  }

  // Rendering the additional "desc" field for arith constraints
  let render_extra_description(constraint) = {
    ([_description_], [], eval(constraint.desc, mode: "markup"), [])
  }

  show figure: set block(breakable: true)
  figure(table(
    columns: (auto, auto, 1fr, auto),
    inset: 6pt,
    align: (top + left, top + left, top + left, top + center),
    stroke: none,
    table.header([*Tag*], [*Range*], [*Description*], [*Multiplicity*]),
    table.hline(stroke: stroke(thickness: 2pt)),
    ..for group in groups {
      for constraint in chip.constraints.at(group) {
        (
          [#tag(constraint, lookup_group(group))],
          [#interval(constraint)],
          [#repr_constraint(constraint)],
          [#expr_to_math(constraint.at("multiplicity", default: ""))],
        )
        if has_extra_description(constraint) {
          render_extra_description(constraint)
        }
        if has_polynomial_constraints(constraint) {
          render_polynomial_constraints(constraint)
        }
      }
    },
  ), caption: [Constraint overview of #chip.name chip.])
}
