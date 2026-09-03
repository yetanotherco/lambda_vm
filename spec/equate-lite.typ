// Hacks to make equate work (for our purposes) with HTML export
// Multiple parts copied and reworked from the equate source.
// For paged export, we just fall back to regular equate

#import "@preview/equate:0.3.2": equate as equate-paged

// Mostly copied
#let text-for(nums, numbering_, supplement) = {
  let num = numbering(
    if type(numbering) == str {
      // Trim numbering pattern of prefix and suffix characters.
      let counting-symbols = ("1", "a", "A", "i", "I", "一", "壹", "あ", "い", "ア", "イ", "א", "가", "ㄱ", "*", "①", "⓵")
      let prefix-end = val.numbering.codepoints().position(c => c in counting-symbols)
      let suffix-start = val.numbering.codepoints().rev().position(c => c in counting-symbols)
      numbering_.slice(prefix-end, if suffix-start == 0 { none } else { -suffix-start })
    } else {
      numbering_
    },
    ..nums
  )

  if supplement not in ([], none) [#supplement~#num] else [#num]
}

#let equate-ref(it) = {
  if it.element == none { return it }
  if it.element.func() != metadata { return it }

  let val = it.element.value

  let supplement = if it.supplement == auto {
    val.supplement
  } else if type(it.supplement) == function {
    assert(false, message: "Unsupported feature")
  } else {
    it.supplement
  }

  html.a(href: "#" + str(it.target), text-for(val.nums, val.numbering, supplement))
}

#let show-rule() = it => context {
  // Allow a way to make default equations.
  if it.has("label") and it.label == <equate:revoke> {
    return it
  }

  // We assert that we're doing sub-numbering
  state("equate/sub-numbering", false).update(_ => true)

  let body = it.body
  let children = if repr(body.func()) == "sequence" { body.children } else { (body,) }
  let lines = children.split(linebreak()).filter(line => line != ())

  // Indices of lines that contain a label.
  let labelled = lines
    .enumerate()
    .filter(((i, line)) => {
      if line.len() == 0 { return false }
      if line.last().func() != raw { return false }
      if line.last().lang != "typc" { return false }
      if line.last().text.match(regex("^<.+>$")) == none { return false }
      return true
    })
    .map(((i, _)) => i)

  // Indices of lines that are marked not to be numbered.
  let revoked = lines
    .enumerate()
    .filter(((i, line)) => {
      if i not in labelled { return false }
      return line.last().text == "<equate:revoke>"
    })
    .map(((i, _)) => i)

  // The "revoke" label shall not count as a labelled line.
  labelled = labelled.filter(i => i not in revoked)

  // Indices of numbered lines in this equation.
  let only-outer = labelled.len() == 0 and it.has("label")
  if only-outer {
    assert(revoked.len() == 0, message: "Don't revoke if the outer equation is labelled and no inner is")
  }
  let numbered = if only-outer {
    // We place the outer label halfway
    (lines.len() / 2,)
  } else {
    labelled
  }

  // Main equation number.
  let main-number = counter(math.equation).get()

  let new-lines = (lines
    .enumerate()
    .map(((i, line)) => {
      if i in revoked {
        // Remove "revoke" label and space and return line.
        let _ = if line.at(-2, default: none) == [ ] { line.remove(-2) }
        let _ = line.remove(-1)
        return line
      }

      let (lb, nums) = if i in labelled {
        // Remove trailing spacing (before label).
        let _ = if line.at(-2, default: none) == [ ] { line.remove(-2) }
        // Remove the label
        let lb = line.last().text.slice(1, -1)
        let _ = line.remove(-1)
        (lb, main-number + (numbered.position(n => n == i) + 1,))
      } else if only-outer {
        (str(it.label), main-number)
      } else {
        return line
      }

      let nm = numbering(it.numbering, ..nums)
      line.push($ & $.body)
      line.push(html.elem("mspace", attrs: ("width": "2em")))
      line.push(html.elem("mrow", attrs: ("id": lb, "aria-label": repr(it.supplement).slice(1, -1) + " "  + numbering(it.numbering, ..nums)), nm))
      if not only-outer {
        line.push([#metadata((nums: nums, numbering: it.numbering, supplement: it.supplement))#label(lb)])
      }

      line
    }))

  // Whether the equation is numbered at all.
  let has-numbering = it.numbering != none and type(it.numbering) in (str, function)

  // Whether this equation consumes an equation number.
  let counted = has-numbering and numbered.len() > 0

  // Step the counter only for equations that consume an equation number.
  if not counted {
    counter(math.equation).update(x => x - 1)
  }

  [
    // The equation itself. It is emitted without numbering (the visible
    // number is rendered as text), so that it does not step the counter.
    #math.equation(
      block: true,
      numbering: it.numbering,
      new-lines.join((linebreak(),)).join()
    ) <equate:revoke>

    // The replaced equation steps the counter natively; revert that step.
    #counter(math.equation).update(x => x - 1)
  ]
}

#let equate(
  breakable: auto,
  sub-numbering: false,
  number-mode: "line",
  debug: false,
  body
) = {
  assert(
    sub-numbering == true
    and breakable == true
    and number-mode == "label",
    message: "Unsupported equate-lite variant"
  )

  context {
    if target() == "html" {
      if type(body) == label {
        {
          show ref: equate-ref
          ref(body)
        }
      } else if type(body) == content and body.func() == ref {
        {
          show ref: equate-ref
          body
        }
      } else {
          show math.equation.where(block: true): show-rule()
        body
      }
    } else {
      // Delegate to the original package for paged output.
      show: equate-paged.with(
        breakable: breakable,
        sub-numbering: sub-numbering,
        number-mode: number-mode,
        debug: debug
      )
      body
    }
  }
}
