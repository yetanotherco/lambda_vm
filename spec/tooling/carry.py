"""
This module provides tools to compute the maximum size of the carry limbs in 
mathematical operations on multi-limb integers that are followed by modular 
reduction.
"""

from dataclasses import dataclass
from typing import Tuple

@dataclass
class LimbConfig:
    limb_size: int
    """The size of a limb. Unit: bits"""
    
    nr_limbs: int
    """number of limbs"""        
    
    @property
    def bits(self) -> int:
        """The total number of bits"""
        return self.limb_size * self.nr_limbs
    
    @property
    def max_int(self) -> int:
        """
        The maximum integer that can be represented using `self.bits` bits.
        """
        return (1 << self.bits) - 1
        
    @property
    def limb_modulus(self) -> int:
        return 1 << self.limb_size
    
    @property
    def limb_mask(self) -> int:
        return self.limb_modulus - 1
    
    def limbify(self, x: int) -> "LimbInt":
        return LimbInt(x, self)
       
    

@dataclass
class LimbInt:
    val: int
    limbs: list[int]
    config: LimbConfig

    def __init__(self, x: int | list[int], c: LimbConfig):
        """Given integer `x`, constructs the limb-representation of `x`"""
        assert isinstance(x, (list, int)), f"invalid input type {x=}"

        if isinstance(x, list):
            self.val = self._reconstitute(x, c)
            self.limbs = x
        elif isinstance(x, int):
            self.val = x
            self.limbs = self._limbify(x, c)
        self.config = c
        
        assert self.val <= c.max_int, f"{x=} exceeds maximum int {c.max_int} that can be represented"

    @staticmethod
    def _limbify(x: int, c: LimbConfig) -> list[int]:
        """Convert an integer to a limbified-integer"""
        limbs = []
        while x > 0:
            limbs.append(x & c.limb_mask)
            x >>= c.limb_size
        return limbs + [0] * (c.nr_limbs - len(limbs))
    
    @staticmethod
    def _reconstitute(x: list[int], c: LimbConfig) -> int:
        """Convert a limbified-integer to an integer"""
        return sum(limb * (c.limb_modulus ** i) for i, limb in enumerate(x))
    
    def __repr__(self) -> str:
        repr = "_".join(reversed([f'{L:02X}' for L in self.limbs]))
        return f"""LimbInt({repr})"""
    
    def __mul__(self, o: "LimbInt") -> Tuple["LimbInt", "LimbInt"]:
        assert self.config == o.config, "cannot multiply with different configs"
        
        lhs = self.limbs + [0] * self.config.nr_limbs
        rhs = o.limbs + [0] * self.config.nr_limbs
        
        raw = [
            sum(lhs[j] * rhs[i-j] for j in range(0, i+1))
            for i in range(len(lhs))
        ]
        
        res = []
        carries = []
        c = 0
        for val in raw:
            r = (val + c) % self.config.limb_modulus
            c = (val + c - r) // self.config.limb_modulus
            res.append(r)
            carries.append(c)
        
        double_config = self.config
        double_config.nr_limbs *= 2
        
        res = LimbInt(res, double_config)
        
        return (res, raw, carries)


SECP256K1_P =  0xFFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFE_FFFFFC2F

def formula(L, b, M, A):
    d = 1 if M < A else 0
    return L * M * (2**b - 2) + (L - 1) * M + A - d

if __name__ == "__main__":
    # L16x8Config = LimbConfig(8, 32)
    # p_limbs = L16x8Config.limbify(SECP256K1_P)
    # q_limbs = L16x8Config.limbify(2**256-1)
    # (r, raw, carries) = q_limbs * q_limbs
    # print(f"{p_limbs=}")
    # print(f"{q_limbs=}")
    # print(f"{r=}")
    # print(f"{raw=}")
    # print(f"{carries=}")
    
    print(f"xG² max: { formula(32, 8, 1, 0)}")
    print(f"xG² min: {-formula(32, 8, 1, 1)}")
    
    print(f"yG² max: { formula(32, 8, 1, 0)}")
    print(f"yG² min: {-formula(32, 8, 2, 0)}")
    
    print(f"λ max: { formula(32, 8, 3, 0)}")
    print(f"λ min: {-formula(32, 8, 3, 0)}")
    
    print(f"xR max: { formula(32, 8, 1, 0)}")
    print(f"xR min: {-formula(32, 8, 1, 3)}")
    
    print(f"yR max: { formula(32, 8, 2, 0)}")
    print(f"yR min: {-formula(32, 8, 1, 2)}")
    
    print(f"xG² max: { formula(16, 16, 1, 0)}")
    print(f"xG² min: {-formula(16, 16, 1, 1)}")
    print(f"yG² max: { formula(16, 16, 1, 0)}")
    print(f"yG² min: {-formula(16, 16, 2, 0)}")
    print(f"λ max: { formula(16, 16, 3, 0)}")
    print(f"λ min: {-formula(16, 16, 3, 0)}")
    print(f"xR max: { formula(16, 16, 1, 0)}")
    print(f"xR min: {-formula(16, 16, 1, 3)}")
    print(f"yR max: { formula(16, 16, 2, 0)}")
    print(f"yR min: {-formula(16, 16, 1, 2)}")
