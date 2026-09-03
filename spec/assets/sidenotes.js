// Progressive enhancement: display footnotes as sidenotes in the right margin.
// If no JS available, stick to the regular endnotes.
//
// Most position is done by CSS (sidenotes.css), we just have to insert an aside
// in the right place.
//
// Loaded by a script tag with defer=true so the iife runs at an okay time
// and we avoid DOMContentLoaded stuff
(() => {
  for (const endnote of document.querySelectorAll('section[role="doc-endnotes"] > ol > li')) {
    for (const ref of document.querySelectorAll(`sup[role="doc-noteref"] > a[href="#${CSS.escape(endnote.id)}"]`)) {
      const sup = ref.parentElement;
      const aside = document.createElement("aside");
      aside.className = "sidenote";

      const content = endnote.cloneNode(true);
      aside.replaceChildren(...content.childNodes);

      // Duplicate of the real endnote: keep it out of the tab order and a11y tree
      aside.setAttribute("aria-hidden", "true");
      // incomplete selector, but could be extended if anything becomes relevant, ever
      aside.querySelectorAll("a, summary, iframe, [tabindex]").forEach(el => el.setAttribute("tabindex", "-1"));

      sup.after(aside);
    }
  }
})();
