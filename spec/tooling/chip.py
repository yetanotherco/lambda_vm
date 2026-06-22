import contextlib
import copy
import sys
import tomllib
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Never, Optional, Self


class ErrorReporter:
    reported: bool
    location: list[str]

    def __init__(self, location: str):
        self.reported = False
        self.location = [location]

    def update_location(self, loc: str):
        self.reported = False
        self.location = [loc]

    @contextlib.contextmanager
    def context(self, ctx: str):
        self.location.append(ctx)
        yield
        self.location.pop()

    def error(self, message: str):
        self.reported = True
        print(f"ERROR {'/'.join(self.location)}: {message}", file=sys.stderr)

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

    @classmethod
    def const(cls, x: int) -> Self:
        return cls(x, x)

    def is_bool(self):
        return self.low >= 0 and self.high <= 1

    def is_const(self):
        return self.low == self.high

    def get_const(self) -> int:
        assert self.is_const()
        return self.low


type Type = list[Type] | Range

DEFAULT_TYPE: Type = Range.const(0)


def structure_matches(a: Type, b: Type) -> bool:
    if isinstance(a, Range) and isinstance(b, (Range, type(None))):
        return True
    elif isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(structure_matches(x, y) for x, y in zip(a, b))
    else:
        return False


def constant_fits(cst: int, target: Type) -> bool:
    if isinstance(target, Range):
        return target.low <= cst <= target.high
    else:
        return constant_fits(cst, target[0])


type Expr = (
    LitExpr
    | VarExpr
    | ArrExpr
    | IdxExpr
    | CastExpr
    | MulExpr
    | AddExpr
    | SubExpr
    | ModExpr
    | PowExpr
    | SumExpr
    | NotExpr
    | DummyExpr
)

OPSEL = ["AND", "OR", "XOR", "EQ", "LT", "SHIFT", "SHIFTW", "MUL", "DIVREM"]


@dataclass
class Environment:
    config: "Config"
    valmap: dict[str, Type]
    typemap: dict[str, Type]

    def with_val(self, key: str, val: Range) -> Self:
        return type(self)(self.config, {**self.valmap, key: val}, self.typemap)


@dataclass
class LitExpr:
    lit: int

    def typecheck(self, _env: Environment) -> Type:
        return Range.const(self.lit)


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
class ArrExpr:
    elems: list[Expr]

    def typecheck(self, env: Environment) -> Type:
        reporter.asserts(self.elems != [], f"Empty array: {self!r}")
        return [e.typecheck(env) for e in self.elems]


@dataclass
class IdxExpr:
    base: Expr
    idx: Expr

    def typecheck(self, env: Environment) -> Type:
        base = self.base.typecheck(env)
        idx = self.idx.typecheck(env)
        if not isinstance(idx, Range) or not idx.is_const():
            reporter.error(f"Invalid index: {idx!r}")
            return Range.const(-1)
        idxconst = idx.get_const()
        if isinstance(base, Range):
            reporter.error(f"Indexing into non-array type: {self!r}")
            return DEFAULT_TYPE
        if not (0 <= idxconst < len(base)):
            reporter.error(f"Index out of range {self!r}")
            idxconst = 0
        return base[idxconst]


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
            baselen >= castlen or (isinstance(base, Range) and base.is_const()),
            f"Casting from fewer columns to more: {self!r} {base} {self.type}",
        )
        if isinstance(base, Range) and base.is_const():
            reporter.asserts(
                constant_fits(base.get_const(), self.type),
                f"Casting const to type it doesn't fit: {self!r}",
            )
            if isinstance(self.type, list):
                return [
                    CastExpr(LitExpr(base.get_const() if i == 0 else 0), t).typecheck(env)
                    for i, t in enumerate(self.type)
                ]
            return base
        if isinstance(base, list) and all(b == Range.const(0) for b in base):
            # Workaround for casts of constant zero, to make padding work nicely
            # This may become cleaner if we eventually get to the cast rework from #326
            if isinstance(self.type, Range):
                return Range.const(0)
            else:
                return [CastExpr(LitExpr(0), t).typecheck(env) for t in self.type]
        return self.type


@dataclass
class MulExpr:
    factors: list[Expr]

    def typecheck_binop(self, a: Type, b: Type) -> Type:
        if isinstance(a, list) and isinstance(b, list):
            reporter.error(f"Multiplication of non-scalar types: {self!r}")
            return DEFAULT_TYPE
        elif not isinstance(a, Range):
            return [self.typecheck_binop(x, b) for x in a]
        elif isinstance(b, list):
            return self.typecheck_binop(b, a)
        else:
            extrema = [x * y for x in [a.low, a.high] for y in [b.low, b.high]]
            return Range(min(extrema), max(extrema))

    def typecheck(self, env: Environment) -> Type:
        reporter.asserts(self.factors != [], f"Empty product: {self!r}")
        t: Type = Range.const(1)
        for f in self.factors:
            t = self.typecheck_binop(t, f.typecheck(env))
        return t


@dataclass
class AddExpr:
    terms: list[Expr]

    def typecheck_binop(self, a: Type, b: Type) -> Type:
        if isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                reporter.error(f"Adding array types of different length {self!r}")
                return [DEFAULT_TYPE for _ in b]
            return [self.typecheck_binop(x, y) for x, y in zip(a, b)]
        elif isinstance(a, list) or isinstance(b, list):
            reporter.error(f"Adding of scalar and array types {self!r}")
            return DEFAULT_TYPE
        else:
            return Range(a.low + b.low, a.high + b.high)

    def typecheck(self, env: Environment) -> Type:
        if not self.terms:
            reporter.error("Empty add")
            return Range.const(0)
        t: Type = self.terms[0].typecheck(env)
        for term in self.terms[1:]:
            t = self.typecheck_binop(t, term.typecheck(env))
        return t


@dataclass
class SubExpr:
    head: Expr
    subs: list[Expr]

    def typecheck_binop(self, a: Type, b: Type) -> Type:
        if isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                reporter.error(f"Subtracting array types of different length {self!r}")
                return [DEFAULT_TYPE for _ in a]
            return [self.typecheck_binop(x, y) for x, y in zip(a, b)]
        elif isinstance(a, list) or isinstance(b, list):
            reporter.error(f"Subtraction of scalar and array types {self!r}")
            return DEFAULT_TYPE
        else:
            return Range(a.low - b.high, a.high - b.low)

    def typecheck(self, env: Environment) -> Type:
        t = self.head.typecheck(env)
        if not self.subs:
            if not isinstance(t, Range):
                reporter.error(f"Negating a non-scalar type: {self!r}")
                return t
            return Range(-t.high, -t.low)
        for term in self.subs:
            t = self.typecheck_binop(t, term.typecheck(env))
        return t


@dataclass
class ModExpr:
    elt: Expr
    modulus: Expr

    def typecheck(self, env: Environment) -> Type:
        elt = self.elt.typecheck(env)
        modulus = self.modulus.typecheck(env)

        if isinstance(modulus, list) or not modulus.is_const():
            reporter.error(f"Invalid non-constant modulus: {self.modulus!r}")
            return Range.const(0)
        modulus = modulus.get_const()
        if modulus <= 0:
            reporter.error(f"Invalid non-positive modulus: {self.modulus!r}")
            return Range.const(0)

        if elt.is_const():
            elt = elt.get_const()
            return Range.const(elt % modulus)
        else:
            return Range(0, modulus - 1)


@dataclass
class PowExpr:
    base: Expr
    exp: Expr

    def typecheck(self, env: Environment) -> Type:
        base = self.base.typecheck(env)
        exp = self.exp.typecheck(env)
        if isinstance(base, list) or not base.is_const():
            reporter.error(f"Invalid exponentiation with non-const base: {self.base!r}")
            return DEFAULT_TYPE
        if isinstance(exp, list) or not exp.is_const():
            reporter.error(f"Invalid exponentiation with non-const exponent: {self.exp!r}")
            return DEFAULT_TYPE
        val = pow(base.get_const(), exp.get_const(), env.config.variables.prime)
        return Range.const(val)


@dataclass
class SumExpr:
    iter: "Iter"
    terms: Expr

    def typecheck_binop(self, a: Type, b: Type) -> Type:
        if isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                reporter.error(f"Summing array types of different length {self!r}")
                return [DEFAULT_TYPE for _ in b]
            return [self.typecheck_binop(x, y) for x, y in zip(a, b)]
        elif isinstance(a, list) or isinstance(b, list):
            reporter.error(f"Summing of scalar and array types {self!r}")
            return DEFAULT_TYPE
        else:
            return Range(a.low + b.low, a.high + b.high)

    def typecheck(self, env: Environment) -> Type:
        t: Type = Range.const(0)
        for tc in self.iter.typecheck(env, lambda e: [self.terms.typecheck(e)]):
            t = self.typecheck_binop(t, tc)
        return t


@dataclass
class NotExpr:
    inner: Expr

    def typecheck(self, env: Environment) -> Type:
        inner = self.inner.typecheck(env)
        if isinstance(inner, list) or not inner.is_bool():
            reporter.error(f"Not a bool passed to `not`: {self.inner!r}")
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
            reporter.asserts(x.isidentifier(), f"Invalid identifier name for variable {x!r}")
            return VarExpr(x)
        case ["opsel", str(x)]:
            if x not in OPSEL:
                reporter.error(f"Unknown operation selector: {x!r}")
                return LitExpr(0)
            return LitExpr(OPSEL.index(x))
        case ["arr", *elems]:
            return ArrExpr([build_expr(config, e) for e in elems])
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
            return SubExpr(build_expr(config, head), [build_expr(config, s) for s in subs])
        case ["mod", elt, modulus]:
            return ModExpr(build_expr(config, elt), build_expr(config, modulus))
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


def check_padding_fits(config: "Config", type: Type, pad: Expr):
    def fits(v, t):
        if isinstance(v, Range):
            return v.is_const() and constant_fits(v.get_const(), t)
        else:
            return isinstance(v, list) and isinstance(t, list) and len(v) == len(t) and all(map(fits, v, t))

    val = pad.typecheck(Environment(config, {}, {}))
    reporter.asserts(fits(val, type), f"Invalid padding {pad!r} for type {type!r}")


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
        if isinstance(start, list) or not start.is_const():
            reporter.error(f"Starting value of iterator not a const: {self!r}")
            start = Range.const(0)
        stop = self.stop.typecheck(env)
        if isinstance(stop, list) or not stop.is_const():
            reporter.error(f"Ending value of iterator not a const: {self!r}")
            stop = Range.const(start.get_const())

        # While it's tempting to replace this loop by an assignment of Range(start, stop + 1) to self.name
        # that would break both detection of consts, and narrowing down to the correct type for indexing
        # heterogenous array types
        for i in range(start.get_const(), stop.get_const() + 1):
            yield from callback(env.with_val(self.name, Range.const(i)))


def iters_of(obj: dict, config, name=None) -> list[Iter]:
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
                f"Specified a range on a non-base composite type: {data!r}",
            )
            reporter.asserts(
                isinstance(data["range"], list) and len(data["range"]) == 2,
                f"Invalid range: {data!r}",
            )
            start, stop = data["range"]
            if not isinstance(start, int) and not (isinstance(start, str) and start.isdigit()):
                reporter.error(f"Range start not an int: {data!r}")
                start = 0
            if not isinstance(stop, int) and not (isinstance(stop, str) and stop.isdigit()):
                reporter.error(f"Range end not an int: {data!r}")
                stop = start
            reporter.asserts(int(start) <= int(stop), f"Inverted range: {data!r}")
            self.range = Range(int(start), int(stop))
            self.subtypes = []
        else:
            self.range = None
            self.subtypes = [lookup(tp) for tp in data["subtypes"]]
        self.desc = data["desc"]
        self.preprocessed = data.get("preprocessed", False)

    def as_type(self) -> Type:
        return self.range or self.subtypes[:]


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
        reporter.asserts(
            set(self.instantiated) <= set(self.all),
            f"Instantiated not a subset of all: {self!r}",
        )


@dataclass
class ConfigVariables:
    types: list[TypeConfig]
    categories: ConfigCategories
    prime: int

    def __init__(self, data: dict):
        assert_no_unexpected(data, type(self).__annotations__.keys())
        self.types = []
        base_type = data["types"][0]["label"]
        for tp in data["types"]:
            self.types.append(TypeConfig(base_type, self.lookup_type, tp))
        self.categories = ConfigCategories(data["categories"])
        basefield = self.lookup_type(base_type)
        assert isinstance(basefield, Range)
        self.prime = basefield.high + 1

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
        reporter.asserts(isinstance(self.version, int), f"version {self.version!r} is not an int")


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
    def from_file(cls, filename: str | Path) -> Self:
        reporter.update_location(str(filename))
        with open(filename, "rb") as fp:
            return cls(tomllib.load(fp))

    @classmethod
    def from_string(cls, s: str) -> Self:
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
        with reporter.context(self.name):
            check_padding_fits(config, self.type, self.pad)
        self.precomputed = data.get("precomputed", False)
        reporter.asserts(
            isinstance(self.precomputed, bool),
            f"precomputed is not a bool: {self.precomputed!r}",
        )


def all_iters[T](its: list[Iter], env: Environment, callback: Callable[[Environment], Iterable[T]]) -> Iterable[T]:
    if not its:
        yield from callback(env)
    else:
        yield from its[0].typecheck(env, lambda e: all_iters(its[1:], e, callback))


@dataclass
class PolyWithIters:
    poly: Expr
    iters: list[Iter]


@dataclass
class VirtualDef:
    # A list of polynomials with each a set of iters they range over
    defs: list[PolyWithIters]

    def __init__(self, config: Config, data: dict):
        if "poly" in data:
            idx = data.get("idx", None)
            self.defs = [PolyWithIters(build_expr(config, data["poly"]), iters_of(data, config, name=idx))]
        elif "polys" in data:
            idx = data.get("idx", None)
            self.defs = [
                PolyWithIters(build_expr(config, poly["poly"]), iters_of(poly, config, name=idx))
                for poly in data["polys"]
            ]
        else:
            self.defs = [PolyWithIters(build_expr(config, data), [])]


@dataclass
class VirtualVariable(Variable):
    def_: VirtualDef

    def __init__(self, config: Config, category: str, data: dict):
        assert_no_unexpected(data, (set(Variable.__annotations__.keys()) | {"def"}) - {"pad"})
        reporter.asserts("def" in data, f"Missing def for virtual column: {data!r}")
        data = copy.deepcopy(data)
        def_ = data.pop("def", {})
        super().__init__(config, category, data)
        self.def_ = VirtualDef(config, def_)

    def typecheck(self, env: Environment) -> Type:
        def handle_iters(
            env: Environment,
            iters: list[Iter],
            poly: Expr,
            expected: Type,
            indices: list[int],
            seen: set[tuple],
        ):
            if not iters:
                # Check not doubly defined
                for s in seen:
                    ln = min(len(s), len(indices))
                    if s[:ln] == tuple(indices[:ln]):
                        reporter.error(f"Double definition for virtual column: {self!r} at index {indices}")
                        break

                val = poly.typecheck(env)
                # check val structure matches assigned
                reporter.asserts(
                    structure_matches(val, expected),
                    f"Invalid structure for definition to virtual column: {self!r}",
                )
                # Check type fits?

                seen.add(tuple(indices))
            else:
                it, *its = iters
                # Some duplicated code/concepts from Iter.typecheck
                # But threading the extra needed state through overly complicates everything
                start = it.start.typecheck(env)
                if isinstance(start, list) or not start.is_const():
                    reporter.error(f"Starting value of virtual def iter not a const: {self!r}")
                    start = Range.const(0)
                stop = it.stop.typecheck(env)
                if isinstance(stop, list) or not stop.is_const():
                    reporter.error(f"Ending value of virtual def iter not a const: {self!r}")
                    stop = Range.const(start.get_const())

                if isinstance(expected, Range):
                    reporter.error(f"Virtual definition has an iter for a scalar: {self!r}")
                    return

                if not 0 <= start.get_const() <= stop.get_const() < len(expected):
                    reporter.error(
                        f"Virtual definition index [{start.get_const()}, {stop.get_const()}] out of range for {expected}: {self!r}"
                    )
                    return

                for i in range(start.get_const(), stop.get_const() + 1):
                    handle_iters(
                        env.with_val(it.name, Range.const(i)),
                        its,
                        poly,
                        expected[i],
                        indices + [i],
                        seen,
                    )

        def is_covered(seen: set[tuple], indices: list[int]) -> bool:
            for s in seen:
                if len(s) <= len(indices) and s == tuple(indices[: len(s)]):
                    return True
            return False

        def check_covered(t: Type, seen: set[tuple], indices: list[int]):
            if isinstance(t, Range):
                reporter.asserts(
                    is_covered(seen, indices),
                    f"Virtual column {self.name!r} not completely defined",
                )
            else:
                for i, elt in enumerate(t):
                    check_covered(elt, seen, indices + [i])

        # Special case for better error messages
        if isinstance(self.type, Range):
            reporter.asserts(
                len(self.def_.defs) == 1 and not self.def_.defs[0].iters,
                f"Invalid def for scalar column: {self!r}",
            )
            assigned_type = self.def_.defs[0].poly.typecheck(env)
            if not isinstance(assigned_type, Range):
                reporter.error(f"Assigning non-scalar type to scalar virtual column: {self!r}")
                return self.type
            # Check type fits?
            # Leaving this out because it produces too much noise with one-hot assumptions
            # reporter.asserts(self.type.low <= assigned_type.low <= assigned_type.high <= self.type.high, f"Definition may not fit in virtual column: {self!r}")
        else:
            # Check no indices are covered twice
            seen: set[tuple] = set()
            for poly_iters in self.def_.defs:
                handle_iters(env, poly_iters.iters, poly_iters.poly, self.type, [], seen)
            # Check everything is covered
            check_covered(self.type, seen, [])
        return self.type

    def populate_env(self, env: Environment):
        # We start off general, and assume that the defs
        # are ordered in such a way that each one at most
        # depends on the ones before it
        env.valmap[self.name] = copy.deepcopy(self.type)

        def assign(env, container, its, v):
            idx = env.valmap[its[0].name].get_const()
            if len(its) == 1:
                container[idx] = v
            else:
                assign(env, container[idx], its[1:], v)

        for poly_iters in self.def_.defs:
            if not poly_iters.iters:
                env.valmap[self.name] = poly_iters.poly.typecheck(env)
                continue

            for _ in all_iters(
                poly_iters.iters,
                env,
                lambda e: [assign(e, env.valmap[self.name], poly_iters.iters, poly_iters.poly.typecheck(e))],
            ):
                # Consume the iterator
                pass


@dataclass
class Assumption:
    desc: str
    iters: list[Iter]

    def __init__(self, config: Config, data: dict):
        assert_no_unexpected(data, set(self.__annotations__.keys()) | {"iter", "iters", "ref"})
        self.desc = data["desc"]
        self.iters = iters_of(data, config)


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
        reporter.asserts(
            isinstance(self.constraint, str),
            f"Constraint not a string: {self.constraint!r}",
        )
        self.desc = data.get("desc", "")
        reporter.asserts(isinstance(self.desc, str), f"desc is not a string: {self.desc!r}")
        self.poly = build_expr(config, data["poly"])
        self.iters = iters_of(data, config)

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
                reporter.error(f"Non-scalar value for polynomial constraint: {self!r} {t}")

        for t in all_iters(self.iters, env, lambda e: [self.poly.typecheck(e)]):
            check_includes_zero(t)
        return []


@dataclass
class Signature:
    tag: str
    condition: Optional[Type]
    input: list[Type]
    output: Optional[Type]

    def matches(self, other: Self) -> bool:
        if not isinstance(other, type(self)):
            return False
        if self.tag != other.tag:
            return False
        if (self.output is None) != (other.output is None):
            return False
        if self.output is not None and other.output is not None and not structure_matches(self.output, other.output):
            return False
        # Used as `sig.matches(expected)`, so `self` is the concrete signature found in the toml
        if self.condition is not None and other.condition is None:
            return False
        return structure_matches(self.input, other.input)


@dataclass
class InteractionLike:
    kind: str
    conditional_name: str
    conditional_required: bool
    signature: type[Signature]

    tag: str
    desc: str
    input: list[Expr]
    output: Optional[Expr]
    conditional: Optional[Expr]
    iters: list[Iter]

    def __init__(self, config: Config, data: dict):
        assert_no_unexpected(
            data,
            {
                "tag",
                "desc",
                "input",
                "output",
                self.conditional_name,
                "kind",
                "ref",
                "iter",
                "iters",
            },
        )
        assert data["kind"] == self.kind
        self.tag = data["tag"]
        reporter.asserts(isinstance(self.tag, str), f"tag is not a string: {self.tag!r}")
        self.desc = data.get("desc", "")
        reporter.asserts(isinstance(self.desc, str), f"Description is not a string: {self.desc!r}")
        self.input = [build_expr(config, inp) for inp in data["input"]]
        if "output" in data:
            self.output = build_expr(config, data["output"])
        else:
            self.output = None
        if self.conditional_name in data:
            self.conditional = build_expr(config, data[self.conditional_name])
        else:
            reporter.asserts(
                not self.conditional_required,
                f"Missing {self.conditional_name}: {data!r}",
            )
            self.conditional = None
        self.iters = iters_of(data, config)

    def typecheck(self, env: Environment) -> Iterable[Signature]:
        def callback(e: Environment) -> Iterable[Signature]:
            # TODO: Should we be able to check cond/multiplicity somehow?
            condition = None
            if self.conditional is not None:
                condition = self.conditional.typecheck(e)
            return [
                self.signature(
                    self.tag,
                    condition,
                    [inp.typecheck(e) for inp in self.input],
                    self.output.typecheck(e) if self.output else None,
                )
            ]

        return all_iters(self.iters, env, callback)


class TemplateSignature(Signature):
    pass


class TemplateConstraint(InteractionLike):
    kind = "template"
    conditional_name = "cond"
    conditional_required = False
    signature = TemplateSignature


class InteractionSignature(Signature):
    pass


class InteractionConstraint(InteractionLike):
    kind = "interaction"
    conditional_name = "multiplicity"
    conditional_required = True
    signature = InteractionSignature


@dataclass
class DummyConstraint:
    def typecheck(self, _env: Environment) -> list[Never]:
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
        self.variables = [
            (Variable if cat != "virtual" else VirtualVariable)(config, cat, var)
            for cat, vars in data["variables"].items()
            for var in vars
        ]
        self.assumptions = [Assumption(config, a) for a in data.get("assumptions", [])]
        constraint_groups = [grp["name"] for grp in data.get("constraint_groups", [])]
        assert_no_unexpected(data.get("constraints", {}), constraint_groups)
        self.constraints = [
            build_constraint(config, constraint)
            for group in data.get("constraints", {}).values()
            for constraint in group
        ]

    @classmethod
    def from_file(cls, config: Config, filename: str | Path) -> Self:
        reporter.update_location(str(filename))
        with open(filename, "rb") as fp:
            return cls(config, tomllib.load(fp))

    @classmethod
    def from_string(cls, config: Config, s: str) -> Self:
        reporter.update_location("<string>")
        return cls(config, tomllib.loads(s))

    def typecheck(self) -> Iterable[Signature]:
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
                with reporter.context(v.name):
                    v.typecheck(env)
        for c in self.constraints:
            with reporter.context(repr(c)):
                yield from c.typecheck(env)

    def padding_assignment(self) -> dict[str, Type]:
        env = Environment(self.config, {}, {})
        res = {}
        for v in self.variables:
            if not isinstance(v, VirtualVariable):
                t = v.type
                if isinstance(t, list) and len(t) == 1:
                    t = t[0]
                res[v.name] = CastExpr(v.pad, t).typecheck(env)
        return res

    def check_assignment(
        self,
        chip_and_assigner_for_tag: dict[str, tuple[Self, "SigAssigner"]],
        values: dict[str, Type],
    ):
        reporter.asserts(
            set(values.keys()) <= set(v.name for v in self.variables if not isinstance(v, VirtualVariable)),
            f"Passing unrecognized variable to `check_assignment` of chip {self.name!r}",
        )
        env = Environment(self.config, {}, {})
        for v in self.variables:
            if not isinstance(v, VirtualVariable):
                if v.name not in values:
                    reporter.error(f"Unable to find variable name {v.name!r} when checking assignment")
                    return
                env.valmap[v.name] = values[v.name]
        for v in self.variables:
            if isinstance(v, VirtualVariable):
                v.populate_env(env)

        for c in self.constraints:
            for sig in c.typecheck(env):
                # Recurse on templates
                if isinstance(sig, TemplateSignature) and sig.tag in chip_and_assigner_for_tag:
                    with reporter.context(repr(c)):
                        template, assigner = chip_and_assigner_for_tag[sig.tag]
                        template.check_assignment(chip_and_assigner_for_tag, assigner(sig))


def build_signature(config: Config, data: dict) -> Signature:
    assert_no_unexpected(data, {"tag", "kind", "input", "output", "cond"})
    Sig: type[Signature]
    cond: Optional[Type] = None
    match data["kind"]:
        case "template":
            if "cond" in data:
                cond = build_type(config, data["cond"])
            Sig = TemplateSignature
        case "interaction":
            reporter.asserts("cond" not in data, f"Template signature with cond: {data!r}")
            cond = Range.const(1)
            Sig = InteractionSignature
        case other:
            reporter.error(f"Signature of invalid kind '{other}': {data!r}")
            Sig = Signature
    tag = data["tag"]
    reporter.asserts(isinstance(tag, str), f"Signature tag not a string: {tag!r}")
    input = [build_type(config, inp) for inp in data["input"]]
    if "output" in data:
        output = build_type(config, data["output"])
    else:
        output = None
    return Sig(tag, cond, input, output)


def read_signatures(config, filename) -> list[Signature]:
    with open(filename, "rb") as fp:
        data = tomllib.load(fp)
    assert_no_unexpected(data, {"signatures"})
    return [build_signature(config, sig) for sig in data["signatures"]]


def check_signatures(found: Iterable[Signature], expected: list[Signature]):
    for sig in found:
        reporter.asserts(any(sig.matches(exp) for exp in expected), f"Unexpected signature: {sig}")


type SigAssigner = Callable[[Signature], dict[str, Type]]


def sig_to_assignment(sig: Signature, chip: Chip) -> dict[str, Type]:
    input = sig.input[:]
    output = sig.output
    cond = sig.condition
    values = {}
    for v in chip.variables:
        match v.category:
            case "input":
                values[v.name] = input.pop(0)
            case "output":
                if output is None:
                    reporter.error(f"No output available for template output variable {v.name!r}")
                    return {}
                values[v.name] = output
            case "condition":
                values[v.name] = cond if cond else Range.const(1)
            case "virtual":
                pass
            case other:
                reporter.error(f"Cannot check template with variable of category {other!r}")
    return values


if __name__ == "__main__":
    config = Config.from_file(sys.argv[1])
    signatures = read_signatures(config, sys.argv[2])
    if reporter.reported:
        sys.exit(1)

    reported = False
    chips: list[Chip] = []
    chip_and_assigner_for_tag: dict[str, tuple[Chip, SigAssigner]] = {}
    for file in sys.argv[3:]:
        if file in sys.argv[1:3]:
            continue
        chip = Chip.from_file(config, file)
        chips.append(chip)
        chip_and_assigner_for_tag[chip.name] = (chip, partial(sig_to_assignment, chip=chip))
        reported |= reporter.reported
    if reported:
        sys.exit(1)

    if "ADD" in chip_and_assigner_for_tag:
        add, add_assigner = chip_and_assigner_for_tag["ADD"]
        chip_and_assigner_for_tag["SUB"] = (
            add,
            lambda sig: add_assigner(
                TemplateSignature(sig.tag, sig.condition, [sig.input[0], sig.output], sig.input[1])
            ),
        )

    for chip in chips:
        reporter.update_location(f"Chip {chip.name}")
        check_signatures(chip.typecheck(), signatures)
        reported |= reporter.reported
    for chip in chips:
        reporter.update_location(f"Padding {chip.name}")
        chip.check_assignment(chip_and_assigner_for_tag, chip.padding_assignment())
        reported |= reporter.reported
    if reported:
        sys.exit(1)
    else:
        print("No issues were found.")
        sys.exit(0)
