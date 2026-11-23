(() => {
  // Fix 1: explicitly set "stretchy=false" for parentheses that don't need to,
  // so that chrome doesn't mess up the kerning/horizontal padding
  // Related bug report: https://issues.chromium.org/issues/40256468
  function isBig(node) {
    const selector = "mfrac,munder,mover,munderover,mtable,msup,msub,msubsup";
    return !!node.querySelector(selector) || node.matches(selector)
  }

  const open = "([{⌊⌈";
  const close = ")]}⌋⌉";
  for (let mrow of document.querySelectorAll("math mrow")) {
    let stack = [];
    for (let child of mrow.children) {
      if (child.tagName == "mo" && open.indexOf(child.textContent) != -1) {
        stack.push(child, false);
      } else if (child.tagName == "mo" && close.indexOf(child.textContent) != -1) {
        // We intentionally don't match parens, because of things like half open intervals and round
        if (!stack.length) continue;
        let big = stack.pop();
        if (!big) {
          child.setAttribute("stretchy", "false");
          stack.pop().setAttribute("stretchy", "false");
        } else {
          stack.pop();
        }
        if (stack.length) {
          stack.push(stack.pop() || big);
        }
      } else {
        if (!stack.length) continue;
        stack.push(stack.pop() || isBig(child));
      }
    }

    let big = false;
    while (stack.length) {
      big = stack.pop() || big;
      if (!big) {
        stack.pop().setAttribute("stretchy", "false");
      }
    }
  }


  // Fix 2: Replace the combining accent for `dash` (code point 0x0305) with
  // the overline character (code point 0x203E), for two reasons:
  // - So that nested overlines render properly stretched in Firefox
  // - So that chrome renders it centrally and not offset to the left
  function isOverline(c) {
    return c != "" && "\u0305\u203e\u00af".indexOf(c) != -1;
  }
  for (let elem of document.body.querySelectorAll("mover mo:last-child")) {
    if (isOverline(elem.textContent)) {
      elem.textContent = "\u203e";
    }
  }


  // Fix 3: Replace overlines with a border-top CSS rule for browsers that
  // do not stretch the top line properly. (e.g. chromium (and forks like Brave))
  for (let elem of document.body.querySelectorAll("mover mo:last-child")) {
    if (isOverline(elem.textContent) && elem.getBoundingClientRect().width < elem.parentElement.getBoundingClientRect().width * 0.9) {
      elem.style.display = "none";
      elem.parentElement.style.borderTop = "0.06em solid currentColor";
      elem.parentElement.style.paddingTop = "0.12em";
    }
  }
})()
