// Types and array types
// <type> ::= str
//          | [str, int]

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
  } else if type(typ) == str {
    return raw(typ)
  } else {
    assert(false, message: "Unknown format for type: " + repr(typ))
  }
}

// Render a type to math
#let type_to_math(typ) = type_to_code(typ) // The code version looks reasonable enough in math too


// Expression grammar
// <expr> ::= ()                           ; ""
//          | var                          ; str(var)
//          | int                          ; int
//          | ["idx", expr1, expr2]        ; expr1[expr2]
//          | ["not", expr]                ; !expr
//          | ["+", expr1, expr2, ...]     ; expr1 + expr2 + ...
//          | ["sum", expr1, expr2, expr3] ; Σ_expr1^expr2 expr3
//          | ["*", expr1, expr2, ...]     ; expr1 * expr2 * ...
//          | ["/", expr1, expr2]          ; expr1 / expr2
//          | ["^", expr1, expr2]          ; expr1^expr2
//          | ["=", expr1, expr2]          ; expr1 = expr2
//          | ["-", expr]                  ; -expr
//          | ["-", expr1, expr2, ...]     ; expr1 - expr2 - ...
//          | ["cast", expr, type]         ; expr as type
// 
// 
// To limit the number of parentheses that are placed in an expression,
// the formatter passes `pp` (for Parent Precedence) to each recursive subcall,
// and wraps itself in parentheses when `pp < expr.precedence`.

#let PREC = (
  "MIN": -1, // <the most secret heart of any expression>
  "pow": 0,  // ^
  "neg": 1,  // Unary -
  "cast": 2, // cast
  "mul": 3,  // *
  "div": 4,  // /
  "sum": 5,  // Σ
  "not": 6,  // not
  "add": 7,  // +
  "sub": 8,  // -
  "idx": 9,  // []
  "eq": 10,   // = and :=
  "MAX": 11, // <the void outside every expression>
)

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
      (dict.at(expr.at(0), default: (pp, rec, e) => {
        assert(false, message: "Invalid expression: " + repr(e))
      }))(pp, res, expr)
    }
  }
  res.with(PREC.MAX)
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
    "idx": (pp, rec, e) => rec(PREC.MIN, e.at(1)) + `[` + rec(PREC.MAX, e.at(2)) + `]`,
    "not": (pp, rec, e) => cwrap(`1 - ` + rec(PREC.not, e.at(1)), pp < PREC.not),
    "+": (pp, rec, e) => cwrap(e.slice(1).map(rec.with(PREC.add)).join(` + `), pp < PREC.add),
    "*": (pp, rec, e) => cwrap(e.slice(1).map(rec.with(PREC.mul)).join(` ` + sym.dot + ` `), pp < PREC.mul),
    "/": (pp, rec, e) => cwrap(rec(PREC.div, e.at(1)), pp < PREC.div) + ` / ` + rec(PREC.div, e.at(2)),
    "^": (pp, rec, e) => {
      assert(type(e.at(1)) == int and type(e.at(2)) == int, message: "Can only exponentiate constants")
      // technically wrong associativity, but it's a constant
      rec(PREC.pow, e.at(1)) + `^` + rec(PREC.pow, e.at(2))
    },
    "=": (pp, rec, e) => rec(PREC.eq, e.at(1)) + ` = ` + rec(PREC.eq, e.at(2)),
    ":=": (pp, rec, e) => rec(PREC.eq, e.at(1)) + ` := ` + rec(PREC.eq, e.at(2)),
    "-": (pp, rec, e) => {
      if e.len() == 2 {
        // Negation
        cwrap(`-` + rec(PREC.neg, e.at(1)), pp < PREC.neg)
      } else {
        // Subtraction
        cwrap(e.slice(1).map(rec.with(PREC.sub)).join(` - `), pp < PREC.sub)
      }
    },
    "cast": (pp, rec, e) => {
      assert(e.len() == 3, message: "Invalid type cast: " + repr(e))
      cwrap(rec(PREC.cast, e.at(1)) + ` as ` + type_to_code(e.at(2)), pp < PREC.cast)
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
    "idx": (pp, rec, e) => $#rec(PREC.idx, e.at(1))_(#rec(PREC.idx, e.at(2)))$,
    "not": (pp, rec, e) => mwrap($1 - #rec(PREC.not, e.at(1))$, pp < PREC.not),
    "+": (pp, rec, e) => mwrap($#e.slice(1).map(rec.with(PREC.add)).join($+$)$, pp < PREC.add),
    "sum": (pp, rec, e) => {
      assert(e.len() == 4, message: "invalid sum:" + repr(e))
      mwrap(
        $sum_(#rec(PREC.MAX, e.at(1)))^#rec(PREC.MAX, e.at(2)) #rec(if pp <= PREC.sub {PREC.MAX} else {PREC.sum}, e.at(3))$, 
        pp <= PREC.sub
      )
    },
    "*": (pp, rec, e) => mwrap($#e.slice(1).map(rec.with(PREC.mul)).join($dot$)$, pp < PREC.mul),
    "/": (pp, rec, e) => $#rec(PREC.div, e.at(1)) / #rec(PREC.div, e.at(2))$,
    "^": (pp, rec, e) => {
      assert(type(e.at(1)) == int and type(e.at(2)) == int, message: "Can only exponentiate constants")
      $#e.at(1)^#e.at(2)$
    },
    "=": (pp, rec, e) => $#rec(PREC.eq, e.at(1)) = #rec(PREC.eq, e.at(2))$,
    ":=": (pp, rec, e) => $#rec(PREC.eq, e.at(1)) := #rec(PREC.eq, e.at(2))$,
    "-": (pp, rec, e) => {
      if e.len() == 2 {
        // Negation
        mwrap($-#rec(PREC.neg, e.at(1))$, pp < PREC.neg)
      } else {
        // Subtraction
        mwrap($#e.slice(1).map(rec.with(PREC.sub)).join($-$)$, pp < PREC.sub)
      }
    },
    "cast": (pp, rec, e) => {
      assert(e.len() == 3, message: "Invalid type cast: " + repr(e))
      cwrap($#rec(PREC.cast, e.at(1)) colon.double #type_to_math(e.at(2))$, pp < PREC.cast)
    },
  ),
  var: v => if v.len() == 1 { $#v$ } else { $#raw(v)$ },
  num: n => math.equation[#n],
)
