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

/// Fowler-Noll-Vo (FNV) hash function, version 1a
/// Src: https://en.wikipedia.org/wiki/Fowler-Noll-Vo_hash_function
/// 
/// Note: this is a non-cryptographic hash function; it is optimized
/// for speed at the expense of unpredictability.
/// 
/// This implementation operates on two 32-bit limbs, rather than a single 
/// 64-bit limb, since Typst does not support u64s.
#let FNV-1a(bytes) = {
  // FNV_prime := 0x00000100000001B3
  let prime = (0x000001B3, 0x00000100)

  // hash := FNV_offset_basis = 0xCBF29CE484222325
  let hash = (0x84222325, 0xCBF29CE4)
  for b in bytes {
    // hash := hash XOR byte_of_data
    hash.at(0) = hash.at(0).bit-xor(b)

    // hash := hash × FNV_prime
    let lo = hash.at(0) * prime.at(0)
    let hi = hash.at(0) * prime.at(1) + hash.at(1) * prime.at(0)
    
    // Carry result
    let carry = lo.bit-rshift(32)
    let lo = lo.bit-and(0xFFFFFFFF)
    let hi = (hi + carry).bit-and(0xFFFFFFFF)
    hash = (lo, hi)
  }

  hash.map(int.to-bytes).join()
}

/// Converts a byte array to a hexadecimal string
#let bytes-to-hex(bytes) = {
  /// Pads a string with 0s on the left to reach a certain length
  let z-fill(string) = "0" * (2 - string.len()) + string

  array(bytes)
    .map(b => str(b, base: 16))
    .map(z-fill)
    .sum()
}

/// Tag constraints with an identifier
#let _add_constraint_ids(chip) = {

  /// A NON-CRYPTOGRAPHIC hash function.
  let nchf(str) = FNV-1a(bytes(str))
  
  // number of characters in constraint ID
  let CONSTRAINT_ID_CHAR_COUNT = 4;

  /// Digests a variable based on its location and type.
  let digest_variable(chip, group, idx, var) = {
    /// Flatten the type of a variable into a string
    let flatten_vartype(typ) = {
      if type(typ) == array {
        "(" + typ.map(flatten_vartype).join(",") + ")"
      } else {
        str(typ)
      }
    }
    
    let flattened_type = lower(flatten_vartype(var.type))    
    let input = (chip, group, str(idx), flattened_type).join(",")
    let digest = bytes-to-hex(nchf(input))
    digest.slice(0, count: 8)
  }

  // Map variables to their ID
  let variable_to_ID = chip
    .variables
    .pairs()
    .map(((group, variables)) => {
      variables
        .enumerate()
        .map(((idx, var)) => {
          (var.name: digest_variable(chip.name, group, idx, var))
        }).sum(default: (:))
    }).sum(default: (:))

  // replace variable with ID in LISP
  let replace_variable_with_ID(lisp) = {
    if type(lisp) == array {
      "(" + lisp.map(replace_variable_with_ID).join(",") + ")"
    } else {
      variable_to_ID.at(str(lisp), default: str(lisp))
    }
  }

  // Replace variable names with their ID 
  let digestable_constraint(c) = { 
    let CONSTRAINT_CAT_TO_SCOPE = (
      "interaction": ("iter", "input", "output", "multiplicity"),
      "template": ("iter", "input", "output", "cond"),
      "arith": ("iter", "poly")
    )

    assert(c.kind in CONSTRAINT_CAT_TO_SCOPE)
    let id_tagged = CONSTRAINT_CAT_TO_SCOPE
      .at(c.kind)
      .filter(cat => cat in c.keys())
      .map(cat => (str(cat): replace_variable_with_ID(c.at(cat))))
      .sum(default: (:))

    repr(id_tagged)
      .replace("\n", "")
      .replace(" ", "")
  }

  // Map hash digest to ID
  let digest_to_id(hash_bytes) = {
    let CHARS = "123456789ABDEFGHJKLMNPQRSTUVWXYZ".codepoints()
    assert(CHARS.len() == 32, message: "invalid CHARS length")

    assert(hash_bytes.len() >= 8, message: "too few bytes to digest: " + repr(hash_bytes))
    let int = int.from-bytes(hash_bytes.slice(0, count: 8))
    for i in range(CONSTRAINT_ID_CHAR_COUNT) {
      let idx = int.bit-and(31)
      int = int.bit-rshift(5)
      (CHARS.at(idx), )
    }.sum()
  }

  // Add an ID to each constraint
  chip.constraints = chip.at("constraints", default: (:))
    .pairs()
    .map(((group, constraints)) => {
      (
        str(group):
        constraints
          .map(c => {
            c.id = digest_to_id(nchf(digestable_constraint(c)))
            c
          })
      )
    }).sum(default: (:))

  chip
}

/// Load a chip object from file
///
/// - path(str): path to file containing chip data
/// - config: configuration data this chip needs to match with
#let load_chip(path, config) = {
  let chip = toml(path)
  _check_chip(chip, config)
  return _add_constraint_ids(chip)
}
