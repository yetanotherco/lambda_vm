from dataclasses import dataclass
import sys
import tomllib
from typing import Optional, Union

class ErrorReporter:
    def __init__(self, location):
        self.reported = False
        self.location = location

    def update_location(self, loc):
        self.reported = False
        self.location = loc

    def error(self, message):
        self.reported = True
        print(f"ERROR {self.location}: {message}", file=sys.stderr)

    def asserts(self, condition, message):
        if not condition:
            self.error(message)

reporter = ErrorReporter("unknown")

def assert_no_unexpected(data, possible_keys):
    for key in data.keys():
        reporter.asserts(key in possible_keys, f"Unexpected key: {key!r}")

type Expr = (int
            | VarExpr
            | IdxExpr
            | CastExpr
            | MulExpr
            | AddExpr
            | SubExpr
            | PowExpr
            | SumExpr
            | NotExpr
            | DummyExpr
            )

@dataclass
class VarExpr:
    name: str

@dataclass
class IdxExpr:
    base: Expr
    idx: Expr

@dataclass
class CastExpr:
    base: Expr
    type: "Type"

@dataclass
class MulExpr:
    factors: list[Expr]

@dataclass
class AddExpr:
    terms: list[Expr]

@dataclass
class SubExpr:
    head: Expr
    subs: list[Expr]

@dataclass
class PowExpr:
    base: Expr
    exp: Expr

@dataclass
class SumExpr:
    iter: "Iter"
    terms: Expr

@dataclass
class NotExpr:
    inner: Expr

@dataclass
class DummyExpr:
    pass

def build_expr(config: Optional["Config"], data) -> Expr:
    # TODO
    # Does this need config, or do we delay any config-checking to when we use the expr?
    match data:
        case int(x):
            return x
        case str(x):
            reporter.asserts(x.isidentifier(), f"Invalid identifier name for variable {x!r}")
            return x
        case ["idx", x, y]:
            return IdxExpr(build_expr(config, x), build_expr(config, y))
        case ["cast", x, t]:
            assert config is not None
            return CastExpr(build_expr(config, x), Type(config.variables.types, t))
        case ["*", *factors]:
            return MulExpr([build_expr(config, f) for f in factors])
        case ["+", *terms]:
            return AddExpr([build_expr(config, t) for t in terms])
        case ["-", head, *subs]:
            return SubExpr(build_expr(config, head), [build_expr(config, s) for s in subs])
        case ["^", base, exp]:
            return PowExpr(build_expr(config, base), build_expr(config, exp))
        case ["sum", ["=", str(var), start], stop, terms]:
            return SumExpr(Iter(config, var, start, stop), build_expr(config, terms))
        case ["not", e]:
            return NotExpr(build_expr(config, e))
        case other:
            reporter.error(f"Unknown expression: {other!r}")
            return DummyExpr()

@dataclass
class Iter:
    name: str
    start: Expr
    stop: Expr

    def __init__(self, config, name, start, stop):
        self.name = name
        reporter.asserts(isinstance(self.name, str), f"iter name is not a string: {self.name!r}")
        reporter.asserts(self.name.isidentifier(), f"Not a valid identifier: {self.name!r}")
        self.start = build_expr(config, start)
        self.stop = build_expr(config, stop)

def iters_of(obj, name = None) -> list[Iter]:
    """Return a list of iterators needed by `obj`. Taken from `iters` or `iter`.
    Prepend `name` to every iterator, if given.
    Adapted from the corresponding typst implementation."""
    def clean_iter(it):
        arr = it if isinstance(it, list) else [it]
        if name is not None:
            arr = [name] + arr

        if len(arr) == 2:
            # Assume single-element range
            arr.append(arr[-1])

        if len(arr) != 3:
            reporter.error(f"Invalid length iter: {arr!r}")
            return Iter(config, "_", 0, 0)
        return Iter(config, *arr)

    if "iters" in obj:
        reporter.asserts("iter" not in obj, f"Object has both `iters` and `iter`: {obj!r}")
        return [clean_iter(it) for it in obj["iters"]]
    elif "iter" in obj:
        return [clean_iter(obj["iter"])]
    else:
        return []

@dataclass
class Type:
    base: Union["Type", str]
    dimension: Optional[int]

    def __init__(self, valid_types: Optional[list["TypeConfig"]], data):
        match data:
            case str(x):
                reporter.asserts(valid_types is None or x in [tc.label for tc in valid_types], f"Invalid variable type: {x!r}")
                self.base = x
                self.dimension = None
            case [base, int(dim)]:
                self.base = Type(valid_types, base)
                self.dimension = dim
            case other:
                reporter.error(f"Unable to parse type: {other!r}")

@dataclass
class TypeConfig:
    label: str
    subtypes: list[Type]
    desc: str
    preprocessed: bool

    def __init__(self, data, valid_types=None):
        assert_no_unexpected(data, type(self).__annotations__.keys())
        self.label = data["label"]
        self.subtypes = [Type(valid_types, tp) for tp in data["subtypes"]]
        self.desc = data["desc"]
        self.preprocessed = data.get("preprocessed", False)

@dataclass
class ConfigCategories:
    all: list[str]
    instantiated: list[str]

    def __init__(self, data):
        assert_no_unexpected(data, type(self).__annotations__.keys())
        self.all = data["all"]
        self.instantiated = data["instantiated"]
        reporter.asserts(all(isinstance(v, str) for v in self.all), f"Something's not a string: {self.all}")
        reporter.asserts(all(isinstance(v, str) for v in self.instantiated), f"Something's not a string: {self.instantiated}")


@dataclass
class ConfigVariables:
    types: list[TypeConfig]
    categories: ConfigCategories

    def __init__(self, data):
        assert_no_unexpected(data, type(self).__annotations__.keys())
        self.types = []
        for tp in data["types"]:
            if tp["subtypes"] == [tp["label"]]:
                self.types.append(TypeConfig(tp, valid_types=None))
            else:
                self.types.append(TypeConfig(tp, valid_types=self.types))
        self.categories = ConfigCategories(data["categories"])

@dataclass
class ConfigMetadata:
    version: int

    def __init__(self, data):
        assert_no_unexpected(data, type(self).__annotations__.keys())
        self.version = data["version"]
        reporter.asserts(isinstance(self.version, int), f"version {self.version!r} is not an int")

@dataclass
class Config:
    metadata: ConfigMetadata
    variables: ConfigVariables

    def __init__(self, data):
        """Construct a Config from toml-parsed data"""
        assert_no_unexpected(data, type(self).__annotations__.keys())
        self.metadata= ConfigMetadata(data["metadata"])
        self.variables = ConfigVariables(data["variables"])
        
    @classmethod
    def from_file(cls, filename):
        reporter.update_location(filename)
        return cls(tomllib.load(open(filename, "rb")))

    @classmethod
    def from_string(cls, s):
        reporter.update_location("<string>")
        return cls(tomllib.loads(s))


@dataclass
class Variable:
    category: str
    name: str
    type: Type
    desc: str
    pad: Expr
    precomputed: bool
    
    def __init__(self, config: Config, category: str, data):
        self.category = category
        assert_no_unexpected(data, Variable.__annotations__.keys())
        self.name = data["name"]
        reporter.asserts(isinstance(self.name, str), f"{self.name!r} is not a string")
        reporter.asserts(self.name.isidentifier(), f"Invalid identifier: {self.name!r}")
        self.type = Type(config.variables.types, data["type"])
        self.desc = data["desc"]
        reporter.asserts(isinstance(self.desc, str), f"{self.desc!r} is not a string")
        self.pad = build_expr(None, data.get("pad", 0))
        self.precomputed = data.get("precomputed", False)
        reporter.asserts(isinstance(self.precomputed, bool), f"precomputed is not a bool: {self.precomputed!r}")

@dataclass
class VirtualDef:
    # A list of polynomials with each a set of iters they range over
    defs: list[tuple[list[Iter], Expr]]

    def __init__(self, config: Config, name: str, tp: Type, data):
        # TODO? More sanity checking the format (or is that duplicating work done in typst already)
        if "poly" in data:
            idx = data.get("idx", None)
            self.defs = [(iters_of(data, name = idx), build_expr(config, data["poly"]))]
        elif "polys" in data:
            idx = data.get("idx", None)
            self.defs = [(iters_of(poly, name = idx), build_expr(config, poly["poly"])) for poly in data["polys"]]
        else:
            self.defs = [([], build_expr(config, data))]

@dataclass
class VirtualVariable(Variable):
    def_: VirtualDef
    
    def __init__(self, config: Config, category: str, data):
        assert_no_unexpected(data, set(Variable.__annotations__.keys()) | {"def"})
        reporter.asserts("def" in data, f"Missing def for virtual column: {data!r}")
        def_ = data.pop("def", {})
        super().__init__(config, category, data)
        self.def_ = VirtualDef(config, self.name, self.type, def_)

@dataclass
class Assumption:
    desc: str
    iters: list[Iter]

    def __init__(self, config: Config, data):
        assert_no_unexpected(data, set(self.__annotations__.keys()) | {"iter", "iters", "ref"})
        self.desc = data["desc"]
        self.iters = iters_of(data)

@dataclass
class ArithConstraint:
    constraint: str
    desc: str
    poly: Expr
    iters: list[Iter]

    def __init__(self, config: Config, data):
        assert_no_unexpected(data, set(self.__annotations__.keys()) | {"kind", "ref", "iter", "iters"})
        assert data["kind"] == "arith"
        self.constraint = data["constraint"]
        reporter.asserts(isinstance(self.constraint, str), f"Constraint not a string: {self.constraint!r}")
        self.desc = data.get("desc", "")
        reporter.asserts(isinstance(self.desc, str), f"desc is not a string: {self.desc!r}")
        self.poly = build_expr(config, data["poly"])
        self.iters = iters_of(data)


@dataclass
class TemplateConstraint:
    tag: str
    desc: str
    input: list[Expr]
    output: Optional[Expr]
    cond: Optional[Expr]
    iters: list[Iter]

    def __init__(self, config: Config, data):
        assert_no_unexpected(data, set(self.__annotations__.keys()) | {"kind", "ref", "iter", "iters"})
        assert data["kind"] == "template"
        self.tag = data["tag"]
        reporter.asserts(isinstance(self.tag, str), f"tag is not a string: {self.tag!r}")
        self.desc = data.get("desc", "")
        reporter.asserts(isinstance(self.desc, str), f"Description is not a string: {self.desc!r}")
        self.input = [build_expr(config, inp) for inp in data["input"]]
        if "output" in data:
            self.output = build_expr(config, data["output"])
        else:
            self.output = None
        if "cond" in data:
            self.cond = build_expr(config, data["cond"])
        else:
            self.cond = None
        self.iters = iters_of(data)

@dataclass
class InteractionConstraint:
    tag: str
    input: list[Expr]
    output: Optional[Expr]
    multiplicity: Expr
    iters: list[Iter]

    def __init__(self, config: Config, data):
        assert data["kind"] == "interaction"
        self.tag = data["tag"]
        reporter.asserts(isinstance(self.tag, str), f"tag {self.tag!r} is not a string")
        self.input = [build_expr(config, inp) for inp in data["input"]]
        if "output" in data:
            self.output = build_expr(config, data["output"])
        else:
            self.output = None
        self.multiplicity = build_expr(config, data["multiplicity"])
        self.iters = iters_of(data)

@dataclass
class DummyConstraint:
    pass

type Constraint = ArithConstraint | TemplateConstraint | InteractionConstraint | DummyConstraint

def build_constraint(config, data) -> Constraint:
    match data["kind"]:
        case "arith":
            return ArithConstraint(config, data)
        case "template":
            return TemplateConstraint(config, data)
        case "interaction":
            return InteractionConstraint(config, data)
        case other:
            reporter.error(f"Unknown constraint kind: {other!r}")
            return DummyConstraint()

@dataclass
class Chip:
    config: Config
    name: str
    variables: list[Variable]
    assumptions: list[Assumption]
    constraints: list[Constraint]
    
    def __init__(self, config: Config, data):
        """Construct a chip from toml-parsed data"""
        assert_no_unexpected(data, set(type(self).__annotations__.keys()) | {"constraint_groups"})
        assert_no_unexpected(data["variables"], config.variables.categories.all)
        self.config = config
        self.name = data["name"]
        reporter.asserts(isinstance(self.name, str), f"name is not a string: {self.name!r}")
        reporter.asserts(self.name.isidentifier(), f"Invalid identifier: {self.name!r}")
        self.variables = [(Variable if cat != "virtual" else VirtualVariable)(config, cat, var) for cat, vars in data["variables"].items() for var in vars]
        self.assumptions = [Assumption(config, asm) for asm in data.get("assumptions", [])]
        constraint_groups = [grp["name"] for grp in data.get("constraint_groups", [])]
        assert_no_unexpected(data.get("constraints", {}), constraint_groups)
        self.constraints = [build_constraint(config, con) for group in data.get("constraints", {}).values() for con in group]
        
    @classmethod
    def from_file(cls, config, filename):
        reporter.update_location(filename)
        return cls(config, tomllib.load(open(filename, "rb")))

    @classmethod
    def from_string(cls, config, s):
        reporter.update_location("<string>")
        return cls(config, tomllib.loads(s))
       

if __name__ == "__main__":
    import pprint
    config = Config.from_file(sys.argv[1])
    if reporter.reported:
        sys.exit(1)
    reported = False
    chips = []
    for file in sys.argv[2:]:
        if file == sys.argv[1]:
            continue
        print("Processing", file)
        chips.append(Chip.from_file(config, file))
        reported = reported or reporter.reported
    if not reported:
        pprint.pprint(chips)
