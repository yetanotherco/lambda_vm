/// Path to the config file.
#let CONFIG_PATH = "src/config.toml"

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

/// Check a chip object for internal consistency.
#let _check_chip(chip, config) = {
  // Check that all variable categories are valid
  for category in chip.variables.keys() {
    assert(category in config.variables.categories.all)
  }

  for var in chip.variables.values().flatten() {
    // Check that all variable types are valid
    assert(
      var.type in config.variables.types.map(type => type.label),
      message: "found invalid var type:" + var.type,
    )
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
