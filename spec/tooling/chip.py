from pathlib import Path
import copy
import sys
import tomllib
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Optional, Union, Never

def Bit_type():
    return Type(None, "Bit")

class ErrorReporter:
    reported: bool
    location: str
    
    def __init__(self, location: str):
        self.reported = False
        self.location = location

    def update_location(self, loc: str):
        self.reported = False
        self.location = loc

    def error(self, message: str):
        self.reported = True
        print(f"ERROR {self.location}: {message}", file=sys.stderr)

    def asserts(self, condition: bool, message: str):
        if not condition:
            self.error(message)

reporter = ErrorReporter("unknown")

def assert_no_unexpected(data: dict, possible_keys: Iterable[str]):
    for key in data.keys():
        reporter.asserts(key in possible_keys, f"Unexpected key: {key!r}")
        
    
type Expr = (LitExpr
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

# We can either have an explicit int literal (or const expression) or a known type
# Returning 0 as dummy value should work in most cases, as constants can be used for
# almost anything. The only exception being indexing.
type TypeCheck = Type | int

@dataclass
class Environment:
    config: "Config"
    valmap: dict[str, int]
    typemap: dict[str, "Type"]

    def resolve_index(self, base: "Type", idx: int) -> TypeCheck:
        if base.dimension is not None:
            if not (0 <= idx < base.dimension):
                reporter.error(f"Index out of range for {base!r}: {idx!r}")
                idx = 0
            if isinstance(base.base, str):
                return Type(None, base.base)
            else:
                return base.base

        assert isinstance(base.base, str), "We somehow made a type that's not an array, but has a non-str base"
        typeconfigs = [tc for tc in self.config.variables.types if tc.label == base.base]
        if len(typeconfigs) != 1:
            reporter.error(f"Unable to resolve type: {base!r}")
            return 0
        typeconfig = typeconfigs[0]
        if not (0 <= idx < len(typeconfig.subtypes)):
            reporter.error(f"Index out of range for {base!r}: {idx!r}")
            idx = 0
        return typeconfig.subtypes[idx]

def type_match(a: TypeCheck, b: TypeCheck, context: str) -> TypeCheck:
    """Check that `a` and `b` are "compatible" TypeCheck values.
    That is, either one of them is a constant, or the type is the same"""
    # TODO: improve here; e.g. by allowing thing to match if their subtype is identical?
    # Then would have to return the subtype to be sure?
    # Maybe break everything down to subtypes, then it's purely structural matching?
    if isinstance(a, int):
        return b
    if isinstance(b, int):
        return a
    reporter.asserts(a == b, f"Type mismatch between {a!r} and {b!r} [{context}]")
    return 0

@dataclass
class LitExpr:
    lit: int

    def typecheck(self, _env: Environment) -> TypeCheck:
        return self.lit

@dataclass
class VarExpr:
    name: str

    def typecheck(self, env: Environment) -> TypeCheck:
        if self.name in env.valmap:
            return env.valmap[self.name]
        if self.name in env.typemap:
            return env.typemap[self.name]
        reporter.error(f"Unknown variable: {self.name!r}")
        return 0

@dataclass
class IdxExpr:
    base: Expr
    idx: Expr

    def typecheck(self, env: Environment) -> TypeCheck:
        base = self.base.typecheck(env)
        idx = self.idx.typecheck(env)
        if isinstance(base, int):
            reporter.error(f"Trying to index a constant value: {self.base!r}")
            return 0
        if not isinstance(idx, int):
            reporter.error(f"Trying to index with a non-constant: {self.idx!r}")
            return 0
        return env.resolve_index(base, idx)

@dataclass
class CastExpr:
    base: Expr
    type: "Type"

    def typecheck(self, env: Environment) -> TypeCheck:
        _base = self.base.typecheck(env)
        # TODO? encode/list valid casts
        return self.type

@dataclass
class MulExpr:
    factors: list[Expr]

    def typecheck(self, env: Environment) -> TypeCheck:
        t: TypeCheck = 0
        for f in self.factors:
            t = type_match(t, f.typecheck(env), repr(self))
        return t

@dataclass
class AddExpr:
    terms: list[Expr]

    def typecheck(self, env: Environment) -> TypeCheck:
        t: TypeCheck = 0
        for term in self.terms:
            t = type_match(t, term.typecheck(env), repr(self))
        return t

@dataclass
class SubExpr:
    head: Expr
    subs: list[Expr]

    def typecheck(self, env: Environment) -> TypeCheck:
        t = self.head.typecheck(env)
        for term in self.subs:
            t = type_match(t, term.typecheck(env), repr(self))
        return t

@dataclass
class PowExpr:
    base: Expr
    exp: Expr

    def typecheck(self, env: Environment) -> TypeCheck:
        base = self.base.typecheck(env)
        exp = self.exp.typecheck(env)
        if not isinstance(base, int):
            reporter.error(f"Invalid exponentiation with non-const base: {self.base!r}")
            return 0
        if not isinstance(exp, int):
            reporter.error(f"Invalid exponentiation with non-const exponent: {self.exp!r}")
            return 0
        return base**exp
            

@dataclass
class SumExpr:
    iter: "Iter"
    terms: Expr

    def typecheck(self, env: Environment) -> TypeCheck:
        t: TypeCheck = 0
        for tc in self.iter.typecheck(env, lambda e: [self.terms.typecheck(e)]):
            t = type_match(t, tc, repr(self))
        return t

@dataclass
class NotExpr:
    inner: Expr

    def typecheck(self, env: Environment) -> TypeCheck:
        inner = self.inner.typecheck(env)
        if isinstance(inner, int):
            reporter.asserts(inner in {0, 1}, f"Not a bool passed to `not`: {self.inner!r}")
            return 1 - inner
        reporter.asserts(inner == Bit_type(), f"Not a bool passed to `not`: {self.inner!r}")
        return Bit_type()

@dataclass
class DummyExpr:
    def typecheck(self, _env: Environment) -> TypeCheck:
        return 0

def build_expr(config: Optional["Config"], data: object) -> Expr:
    # Does this need config, or do we delay any config-checking to when we use the expr?
    match data:
        case int(x):
            return LitExpr(x)
        case str(x):
            reporter.asserts(x.isidentifier(), f"Invalid identifier name for variable {x!r}")
            return VarExpr(x)
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
            assert config is not None
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

    def __init__(self, config: "Config", name: str, start: object, stop: object):
        self.name = name
        reporter.asserts(isinstance(self.name, str), f"iter name is not a string: {self.name!r}")
        reporter.asserts(self.name.isidentifier(), f"Not a valid identifier: {self.name!r}")
        self.start = build_expr(config, start)
        self.stop = build_expr(config, stop)

    def typecheck[T](self, env: Environment, callback: Callable[[Environment], Iterable[T]]) -> Iterable[T]:
        start = self.start.typecheck(env)
        if not isinstance(start, int):
            reporter.error(f"Starting value of summation not a const: {self!r}")
            start = 0
        stop = self.stop.typecheck(env)
        if not isinstance(stop, int):
            reporter.error(f"Ending value of summation not a const: {self!r}")
            stop = 0

        for i in range(start, stop + 1):
            old_env = copy.deepcopy(env)
            env.valmap[self.name] = i
            yield from callback(env)
            env = old_env

def iters_of(obj: dict, name = None) -> list[Iter]:
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

    def __init__(self, valid_types: Optional[list["TypeConfig"]], data: object):
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

    def __init__(self, data: dict, valid_types=None):
        assert_no_unexpected(data, type(self).__annotations__.keys())
        self.label = data["label"]
        self.subtypes = [Type(valid_types, tp) for tp in data["subtypes"]]
        self.desc = data["desc"]
        self.preprocessed = data.get("preprocessed", False)

@dataclass
class ConfigCategories:
    all: list[str]
    instantiated: list[str]

    def __init__(self, data: dict):
        assert_no_unexpected(data, type(self).__annotations__.keys())
        self.all = data["all"]
        self.instantiated = data["instantiated"]
        reporter.asserts(all(isinstance(v, str) for v in self.all), f"Something's not a string: {self.all}")
        reporter.asserts(all(isinstance(v, str) for v in self.instantiated), f"Something's not a string: {self.instantiated}")


@dataclass
class ConfigVariables:
    types: list[TypeConfig]
    categories: ConfigCategories

    def __init__(self, data: dict):
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

    def __init__(self, data: dict):
        assert_no_unexpected(data, type(self).__annotations__.keys())
        self.version = data["version"]
        reporter.asserts(isinstance(self.version, int), f"version {self.version!r} is not an int")

@dataclass
class Config:
    metadata: ConfigMetadata
    variables: ConfigVariables

    def __init__(self, data: dict):
        """Construct a Config from toml-parsed data"""
        assert_no_unexpected(data, type(self).__annotations__.keys())
        self.metadata= ConfigMetadata(data["metadata"])
        self.variables = ConfigVariables(data["variables"])
        
    @classmethod
    def from_file(cls, filename: str | Path) -> "Config":
        reporter.update_location(str(filename))
        return cls(tomllib.load(open(filename, "rb")))

    @classmethod
    def from_string(cls, s: str) -> "Config":
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
    
    def __init__(self, config: Config, category: str, data: dict):
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

def all_iters[T](its: list[Iter], env: Environment, callback: Callable[[Environment], Iterable[T]]) -> Iterable[T]:
    if not its:
        yield from callback(env)
    else:
        yield from its[0].typecheck(env, lambda e: all_iters(its[1:], e, callback))

@dataclass
class VirtualDef:
    # A list of polynomials with each a set of iters they range over
    defs: list[tuple[list[Iter], Expr]]

    def __init__(self, config: Config, name: str, tp: Type, data: dict):
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
    
    def __init__(self, config: Config, category: str, data: dict):
        assert_no_unexpected(data, set(Variable.__annotations__.keys()) | {"def"})
        reporter.asserts("def" in data, f"Missing def for virtual column: {data!r}")
        def_ = data.pop("def", {})
        super().__init__(config, category, data)
        self.def_ = VirtualDef(config, self.name, self.type, def_)

    def typecheck(self, env: Environment) -> TypeCheck:
        # TODO
        return 0

@dataclass
class Assumption:
    desc: str
    iters: list[Iter]

    def __init__(self, config: Config, data: dict):
        assert_no_unexpected(data, set(self.__annotations__.keys()) | {"iter", "iters", "ref"})
        self.desc = data["desc"]
        self.iters = iters_of(data)

@dataclass
class ArithConstraint:
    constraint: str
    desc: str
    poly: Expr
    iters: list[Iter]

    def __init__(self, config: Config, data: dict):
        assert_no_unexpected(data, set(self.__annotations__.keys()) | {"kind", "ref", "iter", "iters"})
        assert data["kind"] == "arith"
        self.constraint = data["constraint"]
        reporter.asserts(isinstance(self.constraint, str), f"Constraint not a string: {self.constraint!r}")
        self.desc = data.get("desc", "")
        reporter.asserts(isinstance(self.desc, str), f"desc is not a string: {self.desc!r}")
        self.poly = build_expr(config, data["poly"])
        self.iters = iters_of(data)

    def typecheck(self, env: Environment) -> Iterable[Never]:
        # TODO: is there any reason to typecheck if something is equatable to 0?
        # Iteration for the side effect of typechecking and reporting errors
        for _ in all_iters(self.iters, env, lambda e: [self.poly.typecheck(e)]):
            pass
        return []

@dataclass
class TemplateSignature:
    tag: str
    input: list[TypeCheck]
    output: Optional[TypeCheck]

@dataclass
class TemplateConstraint:
    tag: str
    desc: str
    input: list[Expr]
    output: Optional[Expr]
    cond: Optional[Expr]
    iters: list[Iter]

    def __init__(self, config: Config, data: dict):
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

    def typecheck(self, env: Environment) -> Iterable[TemplateSignature]:
        def callback(e: Environment) -> Iterable[TemplateSignature]:
            # TODO: Should we be able to check cond somehow?
            if self.cond is not None:
                self.cond.typecheck(e)
            return [TemplateSignature(self.tag,
                                        [inp.typecheck(e) for inp in self.input],
                                        self.output.typecheck(e) if self.output else None)]
        return all_iters(self.iters, env, callback)

@dataclass
class InteractionSignature:
    tag: str
    input: list[TypeCheck]
    output: Optional[TypeCheck]

@dataclass
class InteractionConstraint:
    tag: str
    input: list[Expr]
    output: Optional[Expr]
    multiplicity: Expr
    iters: list[Iter]

    def __init__(self, config: Config, data: dict):
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

    def typecheck(self, env: Environment) -> Iterable[InteractionSignature]:
        def callback(e: Environment) -> Iterable[InteractionSignature]:
            # TODO: Should we be able to check multiplicity somehow?
            self.multiplicity.typecheck(e)
            return [InteractionSignature(self.tag,
                                        [inp.typecheck(e) for inp in self.input],
                                        self.output.typecheck(e) if self.output else None)]
        return all_iters(self.iters, env, callback)

@dataclass
class DummyConstraint:
    def typecheck(self, env: Environment) -> list[Never]:
        return []

type Constraint = ArithConstraint | TemplateConstraint | InteractionConstraint | DummyConstraint

def build_constraint(config, data: dict) -> Constraint:
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
    
    def __init__(self, config: Config, data: dict):
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
    def from_file(cls, config: Config, filename: str | Path) -> "Chip":
        reporter.update_location(str(filename))
        return cls(config, tomllib.load(open(filename, "rb")))

    @classmethod
    def from_string(cls, config: Config, s: str) -> "Chip":
        reporter.update_location("<string>")
        return cls(config, tomllib.loads(s))

    def typecheck(self) -> Iterable[TemplateSignature | InteractionSignature]:
        env = Environment(self.config, {}, {v.name: v.type for v in self.variables})
        for v in self.variables:
            if isinstance(v, VirtualVariable):
                v.typecheck(env)
        for c in self.constraints:
            yield from c.typecheck(env)
       

if __name__ == "__main__":
    config = Config.from_file(sys.argv[1])
    if reporter.reported:
        sys.exit(1)
    reported = False
    chips: list[Chip] = []
    for file in sys.argv[2:]:
        if file == sys.argv[1]:
            continue
        chips.append(Chip.from_file(config, file))
        reported = reported or reporter.reported
    if not reported:
        for chip in chips:
            reporter.update_location(f"Chip {chip.name}")
            # TODO: do something with the signatures
            (list(chip.typecheck()))
