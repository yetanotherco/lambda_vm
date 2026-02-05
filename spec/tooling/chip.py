from pathlib import Path
import sys
import tomllib
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Optional, Never


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


@dataclass(frozen=True)
class Range:
    low: int
    high: int

    def is_bool(self):
        return self.low >= 0 and self.high <= 1

    def is_lit(self):
        return self.low == self.high

    def get_lit(self) -> int:
        assert self.is_lit()
        return self.low


type Type = list[Type] | Range

DEFAULT_TYPE: Type = Range(0, 0)

type Expr = (
    LitExpr
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
class Environment:
    config: "Config"
    valmap: dict[str, Range]
    typemap: dict[str, Type]


@dataclass
class LitExpr:
    lit: int

    def typecheck(self, _env: Environment) -> Type:
        return Range(self.lit, self.lit)


@dataclass
class VarExpr:
    name: str

    def typecheck(self, env: Environment) -> Type:
        if self.name in env.valmap:
            return env.valmap[self.name]
        if self.name in env.typemap:
            return env.typemap[self.name]
        reporter.error(f"Unknown variable: {self.name!r}")
        return DEFAULT_TYPE


@dataclass
class IdxExpr:
    base: Expr
    idx: Expr

    def typecheck(self, env: Environment) -> Type:
        base = self.base.typecheck(env)
        idx = self.idx.typecheck(env)
        if not isinstance(idx, Range) or not idx.is_lit():
            reporter.error(f"Invalid index: {idx!r}")
            return Range(-1, -1)
        idxlit = idx.get_lit()
        if not isinstance(base, list):
            reporter.error(f"Indexing into non-array type: {self!r}")
            return DEFAULT_TYPE
        if not (0 <= idxlit < len(base)):
            reporter.error(f"Index out of range {self!r}")
            idxlit = 0
        return base[idxlit]


@dataclass
class CastExpr:
    base: Expr
    type: Type

    def typecheck(self, env: Environment) -> Type:
        base = self.base.typecheck(env)
        # TODO? Detect more sorts of invalid casts
        baselen = len(base) if isinstance(base, list) else 1
        castlen = len(self.type) if isinstance(self.type, list) else 1
        reporter.asserts(
            baselen >= castlen or (isinstance(base, Range) and base.is_lit()),
            f"Casting from fewer columns to more: {self!r} {base} {self.type}",
        )
        return self.type


@dataclass
class MulExpr:
    factors: list[Expr]

    def type_match(self, a: Type, b: Type) -> Type:
        if isinstance(a, list) and isinstance(b, list):
            reporter.error(f"Multiplication of non-scalar types: {self!r}")
            return DEFAULT_TYPE
        elif isinstance(a, list):
            return [self.type_match(x, b) for x in a]
        elif isinstance(b, list):
            return self.type_match(b, a)
        else:
            extrema = [x * y for x in [a.low, a.high] for y in [b.low, b.high]]
            return Range(min(extrema), max(extrema))

    def typecheck(self, env: Environment) -> Type:
        t: Type = Range(1, 1)
        for f in self.factors:
            t = self.type_match(t, f.typecheck(env))
        return t


@dataclass
class AddExpr:
    terms: list[Expr]

    def type_match(self, a: Type, b: Type) -> Type:
        if isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                assert False
                reporter.error(f"Adding array types of different length {self!r}")
                return [DEFAULT_TYPE for _ in range(len(b))]
            return [self.type_match(x, y) for x, y in zip(a, b)]
        elif isinstance(a, list) or isinstance(b, list):
            reporter.error(f"Adding of scalar and array types {self!r}")
            return DEFAULT_TYPE
        else:
            return Range(a.low + b.low, a.high + b.high)

    def typecheck(self, env: Environment) -> Type:
        if not self.terms:
            reporter.error("Empty add")
            return Range(0, 0)
        t: Type = self.terms[0].typecheck(env)
        for term in self.terms[1:]:
            t = self.type_match(t, term.typecheck(env))
        return t


@dataclass
class SubExpr:
    head: Expr
    subs: list[Expr]

    def type_match(self, a: Type, b: Type) -> Type:
        if isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                reporter.error(f"Subtracting array types of different length {self!r}")
                return [DEFAULT_TYPE for _ in range(len(a))]
            return [self.type_match(x, y) for x, y in zip(a, b)]
        elif isinstance(a, list) or isinstance(b, list):
            reporter.error(f"Subtraction of scalar and array types {self!r}")
            return DEFAULT_TYPE
        else:
            return Range(a.low - b.high, a.high - b.low)

    def typecheck(self, env: Environment) -> Type:
        t = self.head.typecheck(env)
        for term in self.subs:
            t = self.type_match(t, term.typecheck(env))
        return t


@dataclass
class PowExpr:
    base: Expr
    exp: Expr

    def typecheck(self, env: Environment) -> Type:
        base = self.base.typecheck(env)
        exp = self.exp.typecheck(env)
        if isinstance(base, list) or not base.is_lit():
            reporter.error(f"Invalid exponentiation with non-const base: {self.base!r}")
            return DEFAULT_TYPE
        if isinstance(exp, list) or not exp.is_lit():
            reporter.error(
                f"Invalid exponentiation with non-const exponent: {self.exp!r}"
            )
            return DEFAULT_TYPE
        val = base.get_lit() ** exp.get_lit()
        return Range(val, val)


@dataclass
class SumExpr:
    iter: "Iter"
    terms: Expr

    def type_match(self, a: Type, b: Type) -> Type:
        if isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                reporter.error(f"Summing array types of different length {self!r}")
                return [DEFAULT_TYPE for _ in range(len(b))]
            return [self.type_match(x, y) for x, y in zip(a, b)]
        elif isinstance(a, list) or isinstance(b, list):
            reporter.error(f"Summing of scalar and array types {self!r}")
            return DEFAULT_TYPE
        else:
            return Range(a.low + b.low, a.high + b.high)

    def typecheck(self, env: Environment) -> Type:
        t: Type = Range(0, 0)
        for tc in self.iter.typecheck(env, lambda e: [self.terms.typecheck(e)]):
            t = self.type_match(t, tc)
        return t


@dataclass
class NotExpr:
    inner: Expr

    def typecheck(self, env: Environment) -> Type:
        inner = self.inner.typecheck(env)
        if isinstance(inner, list) or not inner.is_bool():
            reporter.asserts(
                inner in {0, 1}, f"Not a bool passed to `not`: {self.inner!r}"
            )
            return Range(0, 1)
        return Range(1 - inner.high, 1 - inner.low)


@dataclass
class DummyExpr:
    def typecheck(self, _env: Environment) -> Type:
        return DEFAULT_TYPE


def build_expr(config: Optional["Config"], data: object) -> Expr:
    # Does this need config, or do we delay any config-checking to when we use the expr?
    match data:
        case int(x):
            return LitExpr(x)
        case str(x):
            reporter.asserts(
                x.isidentifier(), f"Invalid identifier name for variable {x!r}"
            )
            return VarExpr(x)
        case ["idx", x, y]:
            return IdxExpr(build_expr(config, x), build_expr(config, y))
        case ["cast", x, t]:
            assert config is not None
            assert isinstance(t, (list, str))
            return CastExpr(build_expr(config, x), build_type(config, t))
        case ["*", *factors]:
            return MulExpr([build_expr(config, f) for f in factors])
        case ["+", *terms]:
            return AddExpr([build_expr(config, t) for t in terms])
        case ["-", head, *subs]:
            return SubExpr(
                build_expr(config, head), [build_expr(config, s) for s in subs]
            )
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
        reporter.asserts(
            isinstance(self.name, str), f"iter name is not a string: {self.name!r}"
        )
        reporter.asserts(
            self.name.isidentifier(), f"Not a valid identifier: {self.name!r}"
        )
        self.start = build_expr(config, start)
        self.stop = build_expr(config, stop)

    def typecheck[T](
        self, env: Environment, callback: Callable[[Environment], Iterable[T]]
    ) -> Iterable[T]:
        start = self.start.typecheck(env)
        if isinstance(start, list) or not start.is_lit():
            reporter.error(f"Starting value of iterator not a const: {self!r}")
            start = Range(0, 0)
        stop = self.stop.typecheck(env)
        if isinstance(stop, list) or not stop.is_lit():
            reporter.error(f"Ending value of iterator not a const: {self!r}")
            stop = Range(start.get_lit(), start.get_lit())

        # While it's tempting to replace this loop by an assignment of Range(start, stop + 1) to self.name
        # that would break both detection of literals, and narrowing down to the correct type for indexing
        # heterogenous array types
        for i in range(start.get_lit(), stop.get_lit() + 1):
            old_val: Optional[Range] = env.valmap.get(self.name, None)
            env.valmap[self.name] = Range(i, i)
            yield from callback(env)
            env.valmap.pop(self.name)
            if old_val is not None:
                env.valmap[self.name] = old_val


def iters_of(obj: dict, name=None) -> list[Iter]:
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
        reporter.asserts(
            "iter" not in obj, f"Object has both `iters` and `iter`: {obj!r}"
        )
        return [clean_iter(it) for it in obj["iters"]]
    elif "iter" in obj:
        return [clean_iter(obj["iter"])]
    else:
        return []


@dataclass
class TypeConfig:
    label: str
    subtypes: list[Type]
    range: Optional[Range]
    desc: str
    preprocessed: bool

    def __init__(self, default_name: str, lookup: Callable[[str], Type], data: dict):
        assert_no_unexpected(data, type(self).__annotations__.keys())
        self.label = data["label"]
        if "range" in data:
            reporter.asserts(
                data["subtypes"] == [default_name],
                f"Specified a range non a non-base composite type: {data!r}",
            )
            reporter.asserts(
                isinstance(data["range"], list) and len(data["range"]) == 2,
                f"Invalid range: {data!r}",
            )
            start, stop = data["range"]
            if not isinstance(start, int):
                reporter.error(f"Range start not an int: {data!r}")
                start = 0
            if not isinstance(stop, int):
                reporter.error(f"Range end not an int: {data!r}")
                stop = start
            self.range = Range(start, stop)
            self.subtypes = []
        else:
            self.range = None
            self.subtypes = [lookup(tp) for tp in data["subtypes"]]
        self.desc = data["desc"]
        self.preprocessed = data.get("preprocessed", False)

    def as_type(self):
        if self.range is not None:
            return self.range
        return self.subtypes[:]


@dataclass
class ConfigCategories:
    all: list[str]
    instantiated: list[str]

    def __init__(self, data: dict):
        assert_no_unexpected(data, type(self).__annotations__.keys())
        self.all = data["all"]
        self.instantiated = data["instantiated"]
        reporter.asserts(
            all(isinstance(v, str) for v in self.all),
            f"Something's not a string: {self.all}",
        )
        reporter.asserts(
            all(isinstance(v, str) for v in self.instantiated),
            f"Something's not a string: {self.instantiated}",
        )


@dataclass
class ConfigVariables:
    types: list[TypeConfig]
    categories: ConfigCategories

    def __init__(self, data: dict):
        assert_no_unexpected(data, type(self).__annotations__.keys())
        self.types = []
        base_type = None
        for tp in data["types"]:
            if base_type is None:
                base_type = tp["label"]
            self.types.append(TypeConfig(base_type, self.lookup_type, tp))
        self.categories = ConfigCategories(data["categories"])

    def lookup_type(self, typename: str) -> Type:
        matches = [t for t in self.types if t.label == typename]
        if len(matches) != 1:
            reporter.error(f"Couldn't lookup type by name: {typename!r}")
            return DEFAULT_TYPE
        return matches[0].as_type()


@dataclass
class ConfigMetadata:
    version: int

    def __init__(self, data: dict):
        assert_no_unexpected(data, type(self).__annotations__.keys())
        self.version = data["version"]
        reporter.asserts(
            isinstance(self.version, int), f"version {self.version!r} is not an int"
        )


@dataclass
class Config:
    metadata: ConfigMetadata
    variables: ConfigVariables

    def __init__(self, data: dict):
        """Construct a Config from toml-parsed data"""
        assert_no_unexpected(data, type(self).__annotations__.keys())
        self.metadata = ConfigMetadata(data["metadata"])
        self.variables = ConfigVariables(data["variables"])

    @classmethod
    def from_file(cls, filename: str | Path) -> "Config":
        reporter.update_location(str(filename))
        return cls(tomllib.load(open(filename, "rb")))

    @classmethod
    def from_string(cls, s: str) -> "Config":
        reporter.update_location("<string>")
        return cls(tomllib.loads(s))


def build_type(config: Config, data: list | str):
    if isinstance(data, list):
        if len(data) != 2:
            reporter.error(f"Invalid type: {data!r}")
            return DEFAULT_TYPE
        return [build_type(config, data[0]) for _ in range(data[1])]
    else:
        return config.variables.lookup_type(data)


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
        self.type = build_type(config, data["type"])
        self.desc = data["desc"]
        reporter.asserts(isinstance(self.desc, str), f"{self.desc!r} is not a string")
        self.pad = build_expr(None, data.get("pad", 0))
        self.precomputed = data.get("precomputed", False)
        reporter.asserts(
            isinstance(self.precomputed, bool),
            f"precomputed is not a bool: {self.precomputed!r}",
        )


def all_iters[T](
    its: list[Iter], env: Environment, callback: Callable[[Environment], Iterable[T]]
) -> Iterable[T]:
    if not its:
        yield from callback(env)
    else:
        yield from its[0].typecheck(env, lambda e: all_iters(its[1:], e, callback))


@dataclass
class VirtualDef:
    # A list of polynomials with each a set of iters they range over
    defs: list[tuple[list[Iter], Expr]]

    def __init__(self, config: Config, name: str, tp: Type, data: dict):
        if "poly" in data:
            idx = data.get("idx", None)
            self.defs = [(iters_of(data, name=idx), build_expr(config, data["poly"]))]
        elif "polys" in data:
            idx = data.get("idx", None)
            self.defs = [
                (iters_of(poly, name=idx), build_expr(config, poly["poly"]))
                for poly in data["polys"]
            ]
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

    def typecheck(self, env: Environment) -> Type:
        def structure_match(a: Type, b: Type):
            if isinstance(a, Range) and isinstance(b, (Range, type(None))):
                return True
            elif isinstance(a, list) and isinstance(b, list):
                return len(a) == len(b) and all(
                    structure_match(x, y) for x, y in zip(a, b)
                )
            else:
                return False

        def handle_iters(
            env: Environment,
            iters: list[Iter],
            poly: Expr,
            expected: Type,
            indices: list[int],
            seen: set[tuple],
        ):
            if not iters:
                asn = poly.typecheck(env)
                # Check not doubly defined
                for s in seen:
                    ln = min(len(s), len(indices))
                    if s[:ln] == tuple(indices[:ln]):
                        reporter.error(
                            f"Double definition for virtual column: {self!r} at index {indices}"
                        )
                        break
                # check asn structure matches assigned
                reporter.asserts(
                    structure_match(asn, expected),
                    f"Invalid structure for definition to virtual column: {self!r}",
                )
                # Check type fits?

                seen.add(tuple(indices))
            else:
                it, *its = iters
                # Some duplicated code/concepts from Iter.typecheck
                start = it.start.typecheck(env)
                if isinstance(start, list) or not start.is_lit():
                    reporter.error(
                        f"Starting value of virtual def iter not a const: {self!r}"
                    )
                    start = Range(0, 0)
                stop = it.stop.typecheck(env)
                if isinstance(stop, list) or not stop.is_lit():
                    reporter.error(
                        f"Ending value of virtual def iter not a const: {self!r}"
                    )
                    stop = Range(start.get_lit(), start.get_lit())

                for i in range(start.get_lit(), stop.get_lit() + 1):
                    if isinstance(expected, Range):
                        reporter.error(
                            f"Virtual definition has an iter for a scalar: {self!r}"
                        )
                        break
                    if not 0 <= i < len(expected):
                        reporter.error(
                            f"Virtual definition index {i} out of range for {expected}: {self!r}"
                        )
                        break
                    old_val: Optional[Range] = env.valmap.get(it.name, None)
                    env.valmap[it.name] = Range(i, i)
                    handle_iters(env, its, poly, expected[i], indices + [i], seen)
                    env.valmap.pop(it.name)
                    if old_val is not None:
                        env.valmap[it.name] = old_val

        def is_covered(seen: set[tuple], indices: list[int]) -> bool:
            for s in seen:
                if len(s) <= len(indices) and s == tuple(indices[:len(s)]):
                    return True
            return False

        def check_covered(t: Type, seen: set[tuple], indices: list[int]):
            if isinstance(t, Range):
                reporter.asserts(is_covered(seen, indices), f"Virtual column {self.name!r} not completely defined")
            else:
                for i in range(len(t)):
                    check_covered(t[i], seen, indices + [i])

        # Special case for better error messages
        if isinstance(self.type, Range):
            reporter.asserts(
                len(self.def_.defs) == 1 and not self.def_.defs[0][0],
                f"Invalid def for scalar column: {self!r}",
            )
            assigned_type = self.def_.defs[0][1].typecheck(env)
            if not isinstance(assigned_type, Range):
                reporter.error(
                    f"Assigning non-scalar type to scalar virtual column: {self!r}"
                )
                return self.type
            # Check type fits?
            # Leaving this out because it produces too much noise with one-hot assumptions
            # reporter.asserts(self.type.low <= assigned_type.low <= assigned_type.high <= self.type.high, f"Definition may not fit in virtual column: {self!r}")
        else:
            # Check no indices are covered twice
            seen: set[tuple] = set()
            for iters, poly in self.def_.defs:
                handle_iters(env, iters, poly, self.type, [], seen)
            # Check everything is covered
            check_covered(self.type, seen, [])
        return self.type


@dataclass
class Assumption:
    desc: str
    iters: list[Iter]

    def __init__(self, config: Config, data: dict):
        assert_no_unexpected(
            data, set(self.__annotations__.keys()) | {"iter", "iters", "ref"}
        )
        self.desc = data["desc"]
        self.iters = iters_of(data)


@dataclass
class ArithConstraint:
    constraint: str
    desc: str
    poly: Expr
    iters: list[Iter]

    def __init__(self, config: Config, data: dict):
        assert_no_unexpected(
            data, set(self.__annotations__.keys()) | {"kind", "ref", "iter", "iters"}
        )
        assert data["kind"] == "arith"
        self.constraint = data["constraint"]
        reporter.asserts(
            isinstance(self.constraint, str),
            f"Constraint not a string: {self.constraint!r}",
        )
        self.desc = data.get("desc", "")
        reporter.asserts(
            isinstance(self.desc, str), f"desc is not a string: {self.desc!r}"
        )
        self.poly = build_expr(config, data["poly"])
        self.iters = iters_of(data)

    def typecheck(self, env: Environment) -> Iterable[Never]:
        # TODO? Should we check that there's no overflow of the modulus?
        #   This would probably struggle due to things like one-hot invariants

        def check_includes_zero(t: Type):
            if isinstance(t, Range):
                reporter.asserts(
                    t.low <= 0 <= t.high,
                    f"Unsatisfiable constraint, 0 not in range: {self!r} {t}",
                )
            else:
                for sub in t:
                    check_includes_zero(sub)

        for t in all_iters(self.iters, env, lambda e: [self.poly.typecheck(e)]):
            check_includes_zero(t)
        return []


@dataclass
class TemplateSignature:
    tag: str
    input: list[Type]
    output: Optional[Type]


@dataclass
class TemplateConstraint:
    tag: str
    desc: str
    input: list[Expr]
    output: Optional[Expr]
    cond: Optional[Expr]
    iters: list[Iter]

    def __init__(self, config: Config, data: dict):
        assert_no_unexpected(
            data, set(self.__annotations__.keys()) | {"kind", "ref", "iter", "iters"}
        )
        assert data["kind"] == "template"
        self.tag = data["tag"]
        reporter.asserts(
            isinstance(self.tag, str), f"tag is not a string: {self.tag!r}"
        )
        self.desc = data.get("desc", "")
        reporter.asserts(
            isinstance(self.desc, str), f"Description is not a string: {self.desc!r}"
        )
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
            return [
                TemplateSignature(
                    self.tag,
                    [inp.typecheck(e) for inp in self.input],
                    self.output.typecheck(e) if self.output else None,
                )
            ]

        return all_iters(self.iters, env, callback)


@dataclass
class InteractionSignature:
    tag: str
    input: list[Type]
    output: Optional[Type]


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
            return [
                InteractionSignature(
                    self.tag,
                    [inp.typecheck(e) for inp in self.input],
                    self.output.typecheck(e) if self.output else None,
                )
            ]

        return all_iters(self.iters, env, callback)


@dataclass
class DummyConstraint:
    def typecheck(self, env: Environment) -> list[Never]:
        return []


type Constraint = (
    ArithConstraint | TemplateConstraint | InteractionConstraint | DummyConstraint
)


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
        assert_no_unexpected(
            data, set(type(self).__annotations__.keys()) | {"constraint_groups"}
        )
        assert_no_unexpected(data["variables"], config.variables.categories.all)
        self.config = config
        self.name = data["name"]
        reporter.asserts(
            isinstance(self.name, str), f"name is not a string: {self.name!r}"
        )
        reporter.asserts(self.name.isidentifier(), f"Invalid identifier: {self.name!r}")
        self.variables = [
            (Variable if cat != "virtual" else VirtualVariable)(config, cat, var)
            for cat, vars in data["variables"].items()
            for var in vars
        ]
        self.assumptions = [
            Assumption(config, asm) for asm in data.get("assumptions", [])
        ]
        constraint_groups = [grp["name"] for grp in data.get("constraint_groups", [])]
        assert_no_unexpected(data.get("constraints", {}), constraint_groups)
        self.constraints = [
            build_constraint(config, con)
            for group in data.get("constraints", {}).values()
            for con in group
        ]

    @classmethod
    def from_file(cls, config: Config, filename: str | Path) -> "Chip":
        reporter.update_location(str(filename))
        return cls(config, tomllib.load(open(filename, "rb")))

    @classmethod
    def from_string(cls, config: Config, s: str) -> "Chip":
        reporter.update_location("<string>")
        return cls(config, tomllib.loads(s))

    def typecheck(self) -> Iterable[TemplateSignature | InteractionSignature]:
        typemap = {}
        for v in self.variables:
            if isinstance(v.type, list) and len(v.type) == 1:
                t = v.type[0]
            else:
                t = v.type
            typemap[v.name] = t

        env = Environment(self.config, {}, typemap)
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
