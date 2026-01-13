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

// Return a list of iterators needed by `obj`. Taken from `iters` or `iter`.
// Prepend `name` to every iterator, if given.
#let iters_of(obj, name: none) = {
  let clean_iter(it) = {
    let arr = if type(it) == array {
      it
    } else {
      (it,)
    }
    if name != none {
      (name,) + arr
    } else {
      arr
    }
  }

  (if "iters" in obj {
    obj.iters
  } else if "iter" in obj {
    (obj.iter,)
  } else {
    ()
  }).map(clean_iter)
}

#let render_chip_padding_table(chip, config) = {
  // Whether `var` is a preprocessed variable.
  let is_preprocessed(var) = {
    config.variables.types
    .filter(t => t.label == var.type)
    .all(t => t.at("preprocessed", default: false))
  }

  let instantiated_vars = config.variables.categories.instantiated.map(c => chip.variables.at(c)).flatten()

  show figure: set block(breakable: true)
  figure(table(
    columns: (auto, auto, auto),
    inset: 6pt,
    align: (right + top, center + top, left + top),
    stroke: none,
    table.header([*Column*], [], [*Padding value*]),
    table.hline(stroke: stroke(thickness: 2pt)),
    ..for var in instantiated_vars {
      if not is_preprocessed(var) {
        ([#raw(var.name)], [$:=$], [#expr_to_math(var.pad)],)
      }
    },
  ), caption: [Overview of padding values for #chip.name chip.])
}

/// Generates a table listing `chip`'s columns.
#let render_chip_column_table(chip, config) = {

  // Render a definition's iterators
  let render_def_iters(iters) = {
    (..for (name, ..args) in iters {
      if args.len() == 1 {
        ([#raw(name) = #expr_to_code(args.at(0))],)
      } else if args.len() == 2 {
        ([#raw(name) #sym.in `[`#expr_to_code(args.at(0)), #expr_to_code(args.at(1))`]`],)
      } else {
        assert(false, message: "Invalid def range: " + repr(name, ..args))
      }
    }).join("\n")
  }

  // Render definition `def`
  let render_definition(def, var_name) = {
    if type(def) in (array, str) {
      return (
        [],
        table.cell(align: right, emph[definition]), 
        table.cell(colspan: 2, expr_to_math(def))
      )
    }

    assert(type(def) == dictionary, message: "invalid definition: " + repr(def))

    let idx = def.at("idx", default: none)
    let gather_indices(obj) = iters_of(obj, name: idx).map(it => it.first())
    let index_all(expr, indices) = {
      for index in indices {
        expr = ("idx", expr, index)
      }
      expr
    }

    if "poly" in def {
      (
        [],
        table.cell(align: right, emph[definition]), 
        expr_to_math((":=", index_all(var_name, gather_indices(def)), def.poly)),
        render_def_iters(iters_of(def, name: idx))
      )
    } else if "polys" in def {
      assert(
        def.polys.map(gather_indices).dedup().len() == 1,
        message: "Can only do multiple polys if they're indexed identically"
      )
      (
        [],
        table.cell(align: right, emph[definition]), 
        table.cell(colspan: 2, expr_to_math(index_all(var_name, gather_indices(def.polys.first()))))
      )
      for (i, poly) in def.polys.enumerate() {
        (
          [],
          [],              
          table.cell(inset: (left: 1.5em), expr_to_math((":=", "", poly.poly))),
          render_def_iters(iters_of(poly, name: idx)),
        )
      }
    } else {
      assert(false, message: "invalid definition: " + repr(def))
    }
  }

  // Group variables by category
  show figure: set block(breakable: true)
  figure(table(
    columns: (auto, auto, 1fr, auto),
    inset: 6pt,
    align: left + top,
    stroke: none,
    table.header([*Label*], [*Type*], table.cell(colspan: 2, [*Description*])),
    table.hline(stroke: stroke(thickness: 2pt)),
    ..for (cat, vars) in chip.variables.pairs() {
      (table.header(level:2, table.cell(colspan: 4, emph(cat))), table.hline(stroke: .6pt))
      for var in vars {
        (
          [#raw(var.name)], 
          [#type_to_code(var.type)], 
          table.cell(colspan: 2, [#eval(var.desc, mode: "markup")])
        )
        if "def" in var {
          render_definition(var.def, var.name)
        }
      }
      (table.cell(colspan: 4, []), )
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

// Render the iterators of `obj`.
#let iters(obj) = {
  iters_of(obj).map(iter => [#raw(iter.at(0)) #sym.in `[`#expr_to_code(iter.at(1)), #expr_to_code(iter.at(2))`]`]).join("\n")
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
    let with_index(x) = ((x,) + iters_of(assumption).map(it => it.at(0))).join(".")
    let lbl = [#chip.name\-A]
    show figure: (it) => align(left, block[#lbl#context with_index(it.counter.display())])
    cref(assumption)[#figure(kind: chip.name + "assumption", numbering: (i) => [#lbl#i], supplement: [], [])]
  }

  figure(table(
    columns: (auto, auto, 1fr),
    inset: 6pt,
    align: (top + left, top + left, top + left),
    stroke: none,
    table.header([*Tag*], [*Range*], [*Description*]),
    table.hline(stroke: stroke(thickness: 2pt)),
    ..for assumption in chip.assumptions {
      ([#tag(assumption)], [#iters(assumption)], [#eval(assumption.desc, mode: "markup")])
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
  let selected_constraints = groups.map(g => ((g): chip.constraints.at(g))).join()

  // Find the group definition in the constraint_groups
  let lookup_group(name) = chip.constraint_groups.filter((g) => g.name == name).at(0, default: (name: name))

  /// Render the contraint's tag.
  let tag(constraint, group) = {
    let with_index(x) = ((x,) + iters_of(constraint).map(it => it.at(0))).join(".")
    let prefix = if "prefix" in group { group.prefix }
    let lbl = [#chip.name\-C#prefix]
    show figure: (it) => align(left, block[#lbl#context with_index(it.counter.display())])
    cref(constraint)[#figure(kind: chip.name + "constraint", numbering: (i) => [#lbl#i], supplement: [], [])]
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
    "desc" in constraint
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
      (table.cell(align: right, colspan: 2, [_polynomial constraint_]), $#expr_to_math(poly) = 0$, [])
    },)
  }

  // Rendering the additional "desc" field for arith constraints
  let render_extra_description(constraint) = {
    (table.cell(align: right, colspan: 2, [_description_]), eval(constraint.desc, mode: "markup"), [])
  }

  // Whether there is at least one constraint with a range
  // This can be used to see whether the "Range" label should be displayed
  let do_display_range = selected_constraints.values().flatten().any(x => iters_of(x).len() > 0)

  // Whether there is at least one constraint with a multiplicity
  // This can be used to see whether the "Multiplicity" label should be displayed
  let do_display_multiplicity = selected_constraints.values().flatten().any(x => "multiplicity" in x)

  show figure: set block(breakable: true)
  figure(table(
    columns: (auto, auto, 1fr, auto),
    inset: 6pt,
    align: (top + left, top + left, top + left, top + center),
    stroke: none,
    table.header(
      [*Tag*], 
      if do_display_range {[*Range*]} else {[]}, 
      [*Description*], 
      if do_display_multiplicity {[*Multiplicity*]} else {[]},
    ),
    table.hline(stroke: stroke(thickness: 2pt)),
    ..for (group, group_constraints) in selected_constraints.pairs() {
      for constraint in group_constraints {
        (
          [#tag(constraint, lookup_group(group))],
          [#iters(constraint)],
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
    }
  ), caption: [Constraint overview of #chip.name chip.])
}
