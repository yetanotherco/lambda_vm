/// Path to the config file.
#let CONFIG_PATH = "src/config.toml"
/// Path to the signatures file
#let SIGNATURES_PATH = "src/signatures.toml"

/// Check the configuration object for internal consistency.
#let _check_config(config) = {
  // Check that variable subtypes are listed, or "none"
  let types = config.variables.types
  for type in types {
    for subtype in type.subtypes {
      assert(
        subtype in types.map(type => type.label),
        message: "subtype '" + subtype + "' does not exist.",
      )
    }
  }

  // Check that `instantiated` variables are a subset of `all`
  let categories = config.variables.categories
  for category in categories.instantiated {
    assert(
      category in categories.all,
      message: "category '" + category + "' part of `instantiated`, but not `all`.",
    )
  }
}

/// Load the configuration file.
#let load_config() = {
  let config = toml(CONFIG_PATH)
  _check_config(config)
  return config
}


// Validate the `signatures` overview
#let _check_signatures(signatures, config) = {
  let var_labels = config.variables.types.map(t => t.label)

  // Verify that `var` is a valid variable.
  let verify_variable(var) = {
    while type(var) == array {
        assert(type(var.at(1)) == int, message: "Invalid var type: " + repr(var))
        var = var.at(0)
    }
    if type(var) == str {
      assert(var in var_labels, message: "Invalid var type: " + repr(var))
    } else {
      assert(false, message: "Invalid var type: " + repr(var))
    }
  }

  assert("signatures" in signatures, message: "No signatures listed")
  for sig in signatures.signatures {
    assert("tag" in sig, message: "No tag associated with " + repr(sig))
    assert(type(sig.tag) == str, message: "Tag is not of type str: " + repr(sig.tag))

    assert("kind" in sig, message: "No kind associated with " + repr(sig))
    assert(type(sig.kind) == str, message: "kind is not of type str: " + repr(sig.kind))
    assert(sig.kind in ("interaction", "template"), message: "Invalid kind: " + repr(sig.kind))

    if "cond" in sig {
      assert(sig.kind != "interaction", message: "Invalid condition for interaction: " + repr(sig))
      verify_variable(sig.cond)      
    }    
    
    assert("input" in sig, message: "No input associated with " + repr(sig))
    assert(type(sig.input) == array, message: "Invalid input type: " + repr(sig.input))
    sig.input.map(i => verify_variable(i))

    if "output" in sig {
      verify_variable(sig.output)
    }
  }
}

// Load the signatures from file
#let load_signatures(config) = {
  let signatures = toml(SIGNATURES_PATH)
  _check_signatures(signatures, config)
  return signatures
}


/// Check a chip object for internal consistency.
#let _check_chip(chip, config) = {
  // Check that all variable categories are valid
  for category in chip.variables.keys() {
    assert(
      category in config.variables.categories.all, 
      message: "invalid category: " + repr(category)
    )
  }

  // Check that `def` is only contained in `virtual` variables
  let non_virtual_vars = chip.variables.pairs().filter(x => x.first() != "virtual").map(x => x.last()).flatten();
  for var in non_virtual_vars {
    assert(
      "def" not in var,
      message: "illegal `def` in non-virtual var: " + repr(var.name)
    )
  }

  let all_vars = chip.variables.values().flatten()
  let all_labels = config.variables.types.map(type => type.label);
  for var in all_vars {
    let type_label = var.type
    while type(type_label) == array {
      assert(type_label.len() == 2 and type(type_label.at(1)) == int, message: "invalid type: " + repr(var.type))
      type_label = type_label.at(0)
    }
    // Check that all variable types are valid
    assert(type_label in all_labels, message: "found invalid var type: " + repr(var.type))
  }
}

/// Load a chip object from file
///
/// - path(str): path to file containing chip data
/// - config: configuration data this chip needs to match with
#let load_chip(path, config) = {
  let chip = toml(path)
  _check_chip(chip, config)
  return chip
}
