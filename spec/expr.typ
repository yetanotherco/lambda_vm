// Grammar
// <expr> ::= ()                           ; ""
//          | var                          ; str(var)
//          | int                          ; int
//          | ["idx", expr1, expr2]        ; expr1[expr2]
//          | ["not", expr]                ; !expr
//          | ["+", expr1, expr2, ...]     ; expr1 + expr2 + ...
//          | ["*", expr1, expr2, ...]     ; expr1 * expr2 * ...
//          | ["/", expr1, expr2]          ; expr1 / expr2
//          | ["^", expr1, expr2]          ; expr1^expr2
//          | ["=", expr1, expr2]          ; expr1 = expr2
//          | ["-", expr]                  ; -expr
//          | ["-", expr1, expr2, ...]     ; expr1 - expr2 - ...
// 
// 
// To limit the number of parentheses that are placed in an expression,
// the formatter passes `pp` (for Parent Precedence) to each recursive subcall,
// and wraps itself in parentheses when `pp < expr.precedence`.
//
// Precedence values:
// 0 : ^
// 1 : neg (e.g., 5 =>  -5)
// 2 : *
// 3 : /
// 4 : not (e.g., 5 => 1-5)
// 5 : +
// 6 : -
// 7 : []
// 8 : =
// 10: <the void outside every expression>
#let MAX_PRECEDENCE = 10

// Mutual recursion through a trick from https://github.com/typst/typst/issues/744
#let make_expr_formatter(dict, empty: none, var: raw, num: str) = {
  let res(pp, expr) = {
    if expr == none {
      empty
    } else if type(expr) == str {
      var(expr)
    } else if type(expr) == int {
      num(expr)
    } else if type(expr) == array {
      (dict.at(expr.at(0), default: (e) => {
        assert(false, "Invalid expression: " + repr(e))
      }))(pp, res, expr)
    }
  }
  res.with(MAX_PRECEDENCE)
}

// Wrap code `expr` if `apply = true`
#let cwrap(expr, apply) = {
  if apply {
    `(` + expr + `)`
  } else {
    expr
  }
}

// Typeset an expression as code
#let expr_to_code = make_expr_formatter(
  (
    "idx": (pp, rec, e) => rec(0, e.at(1)) + `[` + rec(10, e.at(2)) + `]`,
    "not": (pp, rec, e) => cwrap(`1 - ` + rec(4, e.at(1)), pp < 4),
    "+": (pp, rec, e) => cwrap(e.slice(1).map(rec.with(5)).join(` + `), pp < 5),
    "*": (pp, rec, e) => cwrap(e.slice(1).map(rec.with(2)).join(` ` + sym.dot + ` `), pp < 2),
    "/": (pp, rec, e) => cwrap(rec(3, e.at(1)), pp < 3) + ` / ` + rec(3, e.at(2)),
    "^": (pp, rec, e) => {
      assert(type(e.at(1)) == int and type(e.at(2)) == int, message: "Can only exponentiate constants")
      rec(0, e.at(1)) + `^` + rec(0, e.at(2))
    },
    "=": (pp, rec, e) => rec(8, e.at(1)) + ` = ` + rec(8, e.at(2)),
    "-": (pp, rec, e) => {
      if e.len() == 2 {
        // Negation
        cwrap(`-` + rec(1, e.at(1)), pp < 1)
      } else {
        // Subtraction
        cwrap(e.slice(1).map(rec.with(6)).join(` - `), pp < 6)
      }
    },
  ),
)

// Wrap math `expr` if `apply = true`
#let mwrap(expr, apply) = {
  if apply {
    $($ + expr + $)$
  } else {
    expr
  }
}

// Typeset an expression as math
#let expr_to_math = make_expr_formatter(
  (
    "idx": (pp, rec, e) => $#rec(7, e.at(1))_(#rec(7, e.at(2)))$,
    "not": (pp, rec, e) => mwrap($1 - #rec(4, e.at(1))$, pp < 4),
    "+": (pp, rec, e) => mwrap($#e.slice(1).map(rec.with(5)).join($+$)$, pp < 5),
    "*": (pp, rec, e) => mwrap($#e.slice(1).map(rec.with(3)).join($dot$)$, pp < 3),
    "/": (pp, rec, e) => $#rec(3, e.at(1)) / #rec(3, e.at(2))$,
    "^": (pp, rec, e) => {
      assert(type(e.at(1)) == int and type(e.at(2)) == int, message: "Can only exponentiate constants")
      $#e.at(1)^#e.at(2)$
    },
    "=": (pp, rec, e) => $#rec(8, e.at(1)) = #rec(8, e.at(2))$,
    "-": (pp, rec, e) => {
      if e.len() == 2 {
        // Negation
        mwrap($-#rec(1, e.at(1))$, pp < 1)
      } else {
        // Subtraction
        mwrap($#e.slice(1).map(rec.with(6)).join($-$)$, pp < 6)
      }
    },
  ),
  var: v => if v.len() == 1 { $#v$ } else { $#raw(v)$ },
  num: n => math.equation[#n],
)

// Check that a type expression is structurally valid, without validating against a set of known base types
#let check_array_type(typ) = {
  assert(type(typ.at(0)) == str, message: "Array types need to have a regular type as base")
  assert(type(typ.at(1)) == int, message: "Array types need to have a constant dimension")
}

// Render a type to code
#let type_to_code(typ) = {
  if type(typ) == array {
    check_array_type(typ)
    return raw(typ.at(0) + "[" + str(typ.at(1)) + "]")
  } else if type(typ) == string {
    return raw(typ)
  } else {
    assert(false, message: "Unknown format for type: " + repr(typ))
  }
}

// Render a type to math
#let type_to_math(typ) = render_type_to_code(typ) // The code version looks reasonable enough in math too
