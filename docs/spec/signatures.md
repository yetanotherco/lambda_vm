# Signatures

// Render a signature

let (lb, rb) = if sig.kind == "interaction" { (`[`, `]`) } else if sig.kind == "template" { (`<`, `>`) }

let cond = sig.at("cond", default: none) let cond_str = if cond != none { raw(cond) + ` => ` } else {``}

let input_str = sig.input.map(elt => { if type(elt) == array { raw(elt.at(0)) + `[` + raw(str(elt.at(1))) + `]` } else { raw(elt) } }).join(`, `)

let output = sig.at("output", default: none) let output_str = if output != none { if type(output) == array { raw(output.at(0)) + `[` + raw(str(output.at(1))) + `]` } else { raw(output) } + `; ` } else {``}

return [] }

// Compute the bus size of an interaction

let vars = sig.input + if "output" in sig { (sig.output, )} else {()}

return vars.map(v => { let (label, factor) = if type(v) == array { (v.at(0), v.at(1)) } else { (v, 1) } config.variables.types.filter(type => type.label == label).first().subtypes.len() * factor }) .sum() }

The following lists signatures of the .len() interactions in this VM.

table( columns: (1fr, auto), inset: 7pt, align: (top+left, center), stroke: none, table.header([*Signature*], [*Bus size*]), table.hline(stroke: 1pt), table.vline(stroke: 1pt, x: 1), ..for sig in interactions { ([], []) }, ), caption: "Signature overview of interactions",

Below, we list the signatures of the .len() templates in this VM.

table( columns: 1fr, inset: 7pt, align: (top+left, center), stroke: none, table.header([*Signature*]), table.hline(stroke: 1pt), ..for sig in templates { ([], ) }, ), caption: "Signature overview of templates",