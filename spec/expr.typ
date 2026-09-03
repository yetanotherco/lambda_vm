// Types and array types
// <type> ::= str
//          | [<type>, int]

// Check that a type expression is structurally valid, without validating against a set of known base types
#let check_array_type(typ) = {
  while type(typ) == array {
    assert(typ.len() == 2, message: "Array types must specify two parameters")
    assert(type(typ.at(1)) == int, message: "Array types need to have a constant dimension")
    typ = typ.at(0)
  }
  assert(type(typ) == str, message: "Array types need to have a regular type as base")
}

// Render a type to code
#let type_to_code(typ) = {
  let label = ""
  while type(typ) == array {
    label += "[" + str(typ.at(1)) + "]"
    typ = typ.at(0)
  }
  if type(typ) == str {
    return raw(typ + label)
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
//          | ["arr", expr, ...]           ; [expr, ...]
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
  "idx": 0,  // []
  "pow": 1,  // ^
  "neg": 2,  // Unary -
  "cast": 3, // cast
  "mul": 4,  // *
  "div": 5,  // /
  "mod": 6,  // mod
  "sum": 7,  // Σ
  "not": 8,  // not
  "sub": 9,  // -
  "add": 10,  // +  
  "eq": 11,   // = and :=
  "MAX": 12, // <the void outside every expression>
)

// Mutual recursion through a trick from https://github.com/typst/typst/issues/744
#let make_expr_formatter(dict, empty: none, var: raw, num: str, flatten: (x) => x) = {
  let res(pp, expr) = {
    if expr == none {
      empty
    } else if type(expr) == str {
      var(expr)
    } else if type(expr) == int {
      num(expr)
    } else if type(expr) == array {
      flatten((dict.at(expr.at(0), default: (pp, rec, e) => {
        assert(false, message: "Invalid expression: " + repr(e))
      }))(pp, res, expr))
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

#let flatten_code(x) = {
  if type(x) == array {
    raw(x.map(c => flatten_code(c).text).join(""))
  } else if x.has("children") {
    flatten_code(x.children)
  } else {
    x
  }
}

// Typeset an expression as code
#let expr_to_code = make_expr_formatter(
  (
    "opsel": (pp, rec, e) => {
      assert(type(e.at(1)) == type(""), message: "opsel expects a string")
      `⧼` + raw(e.at(1)) + `⧽`
    },
    "arr": (pp, rec, e) => `[` + e.slice(1).map(rec.with(PREC.MAX)).join(`, `) + `]`,
    "idx": (pp, rec, e) => rec(PREC.MIN, e.at(1)) + `[` + rec(PREC.MAX, e.at(2)) + `]`,
    "not": (pp, rec, e) => cwrap(rec(PREC.not, 1) + ` - ` + rec(PREC.not, e.at(1)), pp < PREC.not),
    "+": (pp, rec, e) => cwrap(e.slice(1).map(rec.with(PREC.add)).join(` + `), pp < PREC.add),
    "sum": (pp, rec, e) => assert(false, message: "sum is unsupported in code."),
    "mod": (pp, rec, e) => {
      assert(e.len() == 3 and type(e.at(2)) == int, message: "Invalid mod expr: " + repr(e))
      cwrap(
        rec(PREC.mod, e.at(1)) + ` % ` + rec(PREC.mod, e.at(2)), 
        pp <= PREC.mod
      ) 
    },
    "*": (pp, rec, e) => {
      if e.len() == 3 and type(e.at(1)) == int and type(e.at(2)) == str and e.at(2).len() == 1 {
        // multiplication of a constant with one-letter variable. 
        // Dropping the "dot"
        cwrap(e.slice(1).map(rec.with(PREC.mul)).join(``), pp < PREC.mul)
      } else {
        cwrap(e.slice(1).map(rec.with(PREC.mul)).join(` ` + sym.dot + ` `), pp < PREC.mul)
      }
    },
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
        cwrap(e.slice(1).map(rec.with(PREC.sub)).join(` - `), pp <= PREC.sub)
      }
    },
    "cast": (pp, rec, e) => {
      assert(e.len() == 3, message: "Invalid type cast: " + repr(e))
      cwrap(rec(PREC.cast, e.at(1)) + ` as ` + type_to_code(e.at(2)), pp < PREC.cast)
    },
  ),
  num: (n) => raw(str(n)),
  flatten: flatten_code
)

// Wrap math `expr` if `apply = true`
#let mwrap(expr, apply) = {
  if apply {
    $($ + expr + $)$
  } else {
    expr
  }
}

#let flat_idxs(expr) = {
  if expr.at(0) != "idx" {
    (expr, ())
  } else {
    let (sub, gathered) = flat_idxs(expr.at(1))
    (sub, gathered + (expr.at(2),))
  }
}

// Typeset an expression as math
#let expr_to_math = make_expr_formatter(
  (
    "opsel": (pp, rec, e) => {
      assert(type(e.at(1)) == type(""), message: "opsel expects a string")
      $lr(chevron.l.curly#raw(e.at(1))chevron.r.curly)$
    },
    "arr": (pp, rec, e) => $[#e.slice(1).map(rec.with(PREC.MAX)).join($, $)]$,
    "idx": (pp, rec, e) => {
      let (val, idxs) = flat_idxs(e)
      $#rec(PREC.idx, val)_(#idxs.map(idx => rec(PREC.idx, idx)).join($, $))$
    },
    "not": (pp, rec, e) => mwrap(rec(PREC.not, 1) + $ - #rec(PREC.not, e.at(1))$, pp < PREC.not),
    "+": (pp, rec, e) => mwrap($#e.slice(1).map(rec.with(PREC.add)).join($+$)$, pp < PREC.add),
    "sum": (pp, rec, e) => {
      assert(e.len() == 4, message: "invalid sum:" + repr(e))
      mwrap(
        $sum_(#rec(PREC.MAX, e.at(1)))^#rec(PREC.MAX, e.at(2)) #rec(if pp <= PREC.sub {PREC.MAX} else {PREC.sum}, e.at(3))$, 
        pp <= PREC.sub
      )
    },
    "mod": (pp, rec, e) => {
      assert(e.len() == 3 and type(e.at(2)) == int, message: "Invalid mod expr: " + repr(e))
      mwrap(
        $#rec(PREC.mod, e.at(1)) mod #rec(PREC.mod, e.at(2))$, 
        pp <= PREC.mod
      )
    },
    "*": (pp, rec, e) => {
      if e.len() == 3 and type(e.at(1)) == int and type(e.at(2)) == str and e.at(2).len() == 1 {
        // multiplication of a constant with one-letter variable. 
        // Dropping the "dot"
        mwrap($#e.slice(1).map(rec.with(PREC.mul)).join($$)$, pp < PREC.mul)
      } else {
        mwrap($#e.slice(1).map(rec.with(PREC.mul)).join($dot$)$, pp < PREC.mul)
      }
    },
    "/": (pp, rec, e) => $#rec(PREC.div, e.at(1)) / #rec(PREC.div, e.at(2))$,
    "^": (pp, rec, e) => {
      assert(type(e.at(1)) == int, message: "Can only exponentiate constants")
      $#e.at(1)^#rec(PREC.MAX, e.at(2))$
    },
    "=": (pp, rec, e) => $#rec(PREC.eq, e.at(1)) = #rec(PREC.eq, e.at(2))$,
    ":=": (pp, rec, e) => $#rec(PREC.eq, e.at(1)) := #rec(PREC.eq, e.at(2))$,
    "-": (pp, rec, e) => {
      if e.len() == 2 {
        // Negation
        mwrap($-#rec(PREC.neg, e.at(1))$, pp < PREC.neg)
      } else {
        // Subtraction
        mwrap(
          $#rec(PREC.add, e.at(1))-#e.slice(2).map(rec.with(PREC.sub)).join($-$)$,
          pp <= PREC.sub
        )
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
