# KECCAK Accelerator

The  chip applies the keccak permutation `kappa` to a given memory range; other aspects of keccak hashing (such as repeated permutation invocation, input padding and state initialization) fall outside the scope of this accelerator.

This permutation `kappa: FF_2^1600 -> FF_2^1600` operates on 1600 bits and is composed of 24 applications of round-permutation `Lambda: FF_2^1600 times NN -> FF_2^1600`, where the additional parameter is the round constant. `Lambda` is defined as the composition `iota compose chi compose pi compose rho compose theta`, where only `iota` depends on the round constant.

The keccak accelerator comprises two chips: a core chip that interacts with the memory --- loading the input and writing the output, and a round chip that applies the round permutation.

## Core chip

### Columns

The  chip is comprised of  variables that are expressed using  columns and leverages  interaction(s):

### Input

| Name | Type | Description |
|------|------|-------------|
| `timestamp` | `DWordWL` | timestamp at which the permutation is performed |
| `addr` | `DWordBL` | memory address storing the first bit of the state |
| `input_state` | `[['Byte', 8], 5][5]` | state at the start of executing the permutation |

### Output

| Name | Type | Description |
|------|------|-------------|
| `output_state` | `[['Byte', 8], 5][5]` | state after executing the permutation |

### Auxiliary

| Name | Type | Description |
|------|------|-------------|
| `state_ptr` | `['DWordHL', 5][5]` | memory addresses storing the entire state |

### Multiplicity

| Name | Type | Description |
|------|------|-------------|
| `μ` | `Bit` |  |

### Constraints

In this VM, we assign syscall number -2 to the  accelerator. The chip therefore contributes the following interaction to the lookup-argument:

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `KECCAK-C1` | `ECALL[timestamp, (2^64 - 2)::DWordWL]` | -μ |

The address containing the state to be permuted is passed in as argument `A0 = x10`. The following constraints describe that this address is read into `addr` ([keccak:c:read_addr]), from which `state_ptr` --- the collection of pointers to all lanes of the state --- is derived ([keccak:c:state_ptr]). The state is then read into `input_state`, while the `output_state` is written back to the indicated address ([keccak:c:load_store_state]).

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `KECCAK-C2` |  | `MEMW[addr; 1, (2 * 10)::DWordWL, addr, timestamp, 1, 0, 0]` | μ |
| `KECCAK-C3.i` | x ∈ [0, 4], y ∈ [0, 4] | `ADD<state_ptr[x][y]::DWordWL; addr::DWordWL, (8 * (5 * y + x))::DWordWL>` |  |
| `KECCAK-C4.i` | x ∈ [0, 4], y ∈ [0, 4], z ∈ [0, 3] | `IS_HALF[state_ptr[x][y][z]]` | μ |
| `KECCAK-C5.i` | x ∈ [0, 4], y ∈ [0, 4] | `MEMW[input_state[x][y]; 0, state_ptr[x][y]::DWordWL, output_state[x][y], timestamp, 0, 0, 1]` | μ |

Lastly, the input state is pushed to the Keccak-round function, while the output after 24 rounds is taken off the bus:

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `KECCAK-C6` | `KECCAK[timestamp, 0, input_state]` | μ |
| `KECCAK-C7` | `KECCAK[timestamp, 24, output_state]` | -μ |

### Padding

The  table can be padded to the next power of two with the following value assignments:

| Column | Padding value |
|--------|---------------|
| `timestamp` | `0` |
| `addr` | `0` |
| `input_state` | `0` |
| `output_state` | `0` |
| `state_ptr` | `8 * [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14], [15, 16, 17, 18, 19], [20, 21, 22, 23, 24]]` |
| `μ` | `0` |

## Round chip

### Columns

The  chip is comprised of  variables that are expressed using  columns and leverages  interaction(s):

### Input

| Name | Type | Description |
|------|------|-------------|
| `timestamp` | `DWordWL` | timestamp at which the permutation is performed |
| `round` | `BaseField` | index of the permutation round |
| `start` | `[['Byte', 8], 5][5]` | state at the start of executing the permutation |

### Auxiliary

| Name | Type | Description |
|------|------|-------------|
| `Cxz` | `[['Byte', 8], 4][5]` | $xor_(i=0)^(y+2) `start[x,i,z]`$ |
| `Cxz_left` | `['Byte', 8][5]` | the left-rotated component of `rotated_Cxz` |
| `Cxz_right` | `['Bit', 4][5]` | the right-rotated component of `rotated_Cxz` (which is a single bit) |
| `Dxz` | `['Byte', 8][5]` | $`Cxz[`\(`x` - 1) mod 5`,y,z]` xor `rotated_Cxz[`\(`x` + 1) mod 5`,y,z]`$ |
| `theta` | `[['Byte', 8], 5][5]` | $theta(`start`)$, the state after applying $theta$. |
| `rot_left` | `[['Byte', 8], 5][5]` | the left-rotated component of $`theta[x,y]` <<< `rnc`$ |
| `rot_right` | `[['Byte', 8], 5][5]` | the right-rotated component of $`theta[x,y]` <<< `rnc`$ |
| `chi_ANDs` | `[['Byte', 8], 5][5]` | $(`pi[`\(x+1) mod 5`,y,z]` xor 255) times.o `pi[`\(x + 2) mod 5`,y,z]`$ |
| `chi` | `[['Byte', 8], 5][5]` | $(chi compose pi compose rho compose theta)(`start`)$; the state after applying $chi$ |
| `rc` | `Byte[8]` | round constants |
| `iota` | `Byte[8]` | state update following from step $iota$. |

### Virtual

| Name | Type | Description |
|------|------|-------------|
| `rotated_Cxz` | `['Byte', 8][5]` | $`Cxz[x,`3`,z]` <<< 1$ |
| `out` | `[['Byte', 8], 5][5]` | state at the end of executing the permutation |
| `rho` | `[['Byte', 8], 5][5]` | $(rho compose theta)(`start`)$; the state after applying $rho$ |
| `pi` | `[['Byte', 8], 5][5]` | $(pi compose rho compose theta)(`start`)$; the state after applying $pi$ |

**Definition of `rotated_Cxz`:**
```
rotated_Cxz := Cxz_left[x][z] + Cxz_right[x][3]
rotated_Cxz := Cxz_left[x][z]
rotated_Cxz := Cxz_left[x][z] + Cxz_right[x][0]
rotated_Cxz := Cxz_left[x][z]
rotated_Cxz := Cxz_left[x][z] + Cxz_right[x][1]
rotated_Cxz := Cxz_left[x][z]
rotated_Cxz := Cxz_left[x][z] + Cxz_right[x][2]
rotated_Cxz := Cxz_left[x][z]
```

**Definition of `out`:**
```
out := iota[z]
out := chi[x][y][z]
out := chi[x][y][z]
out := chi[x][y][z]
```

**Definition of `rho`:**
```
rho := (1 - rbc[x][y][0]) * (1 - rbc[x][y][1]) * (rot_left[x][y][z] + rot_right[x][y][(z - 2) mod 8]) + rbc[x][y][0] * (1 - rbc[x][y][1]) * (rot_left[x][y][(z - 2) mod 8] + rot_right[x][y][(z - 4) mod 8]) + (1 - rbc[x][y][0]) * rbc[x][y][1] * (rot_left[x][y][(z - 4) mod 8] + rot_right[x][y][(z - 6) mod 8]) + rbc[x][y][0] * rbc[x][y][1] * (rot_left[x][y][(z - 6) mod 8] + rot_right[x][y][z])
```

**Definition of `pi`:**
```
pi := rho[(x + 3 * y) mod 5][x][z]
```

### Multiplicity

| Name | Type | Description |
|------|------|-------------|
| `μ` | `Bit` |  |

`start` contains the state to which the permutation should be applied. Its three-dimensional array mimics the specification's three-dimensional state

and orders the bits as prescribed.

Rho rotates every lane by a rotation offset in `[0, 64)`. These offsets are identical for every round.

We decompose each offset in three components: the lower nibble (4 bits) are represented by `rnc`, while the upper two bits are represented by as `Bit`s in `rbc`. That is, ``rho_offset[x][y]` = `rnc[x][y]` + 16 dot `rbc[x][y][0]` + 32 dot `rbc[x][y][1]``.

### Constraints

The following constraints ensure that `theta` captures the state after applying the first subpermutation of the round-permutation: `theta`. Note here that `Cxz_left` and `Cxz_right` do have to be range-checked; it cannot be assumed that this implicitly follows from [keccak:c:Dxz] combined with `rotated_Cxz`'s definition.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `KECCAK_RND-C1.i` | x ∈ [0, 4], z ∈ [0, 7] | `BYTE_ALU[Cxz[x][0][z]; ⧼XOR⧽, start[x][0][z], start[x][1][z]]` | μ |
| `KECCAK_RND-C2.i` | x ∈ [0, 4], y ∈ [2, 4], z ∈ [0, 7] | `BYTE_ALU[Cxz[x][y - 1][z]; ⧼XOR⧽, Cxz[x][y - 2][z], start[x][y][z]]` | μ |
| `KECCAK_RND-C3.i` | x ∈ [0, 4], z ∈ [0, 3] | `HWSL[[(Cxz_left[x]::DWordHL)[z], Cxz_right[x][z]::Half]; (Cxz[x][3]::DWordHL)[z], 1]` | μ |
| `KECCAK_RND-C4.i` | x ∈ [0, 4], z ∈ [0, 7] | μ ⇒ `IS_BYTE<Cxz_left[x][z]>` |  |
| `KECCAK_RND-C5.i` | x ∈ [0, 4], z ∈ [0, 3] | `IS_BIT<Cxz_right[x][z]>` |  |
| `KECCAK_RND-C6.i` | x ∈ [0, 4], z ∈ [0, 7] | `BYTE_ALU[Dxz[x][z]; ⧼XOR⧽, Cxz[(x - 1) mod 5][3][z], rotated_Cxz[(x + 1) mod 5][z]]` | μ |
| `KECCAK_RND-C7.i` | x ∈ [0, 4], y ∈ [0, 4], z ∈ [0, 7] | `BYTE_ALU[theta[x][y][z]; ⧼XOR⧽, start[x][y][z], Dxz[x][z]]` | μ |

Next, we constrain that `rho` captures the state after applying subpermutation `rho`. Note here as well that `rot_left` and `rot_right` do have to be range-checked; it cannot be assumed that this implicitly follows from later constraints.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `KECCAK_RND-C8.i` | x ∈ [0, 4], y ∈ [0, 4], z ∈ [0, 3] | `HWSL[[(rot_left[x][y]::DWordHL)[z], (rot_right[x][y]::DWordHL)[z]]; (theta[x][y]::DWordHL)[z], rnc[x][y]]` | μ |
| `KECCAK_RND-C9.i` | x ∈ [0, 4], y ∈ [0, 4], z ∈ [0, 7] | μ ⇒ `IS_BYTE<rot_left[x][y][z]>` |  |
| `KECCAK_RND-C10.i` | x ∈ [0, 4], y ∈ [0, 4], z ∈ [0, 7] | μ ⇒ `IS_BYTE<rot_right[x][y][z]>` |  |

Observe that the lane-permutation performed by `pi` is absorbed in `pi`'s definition. The next permutation that is constrained in `chi`:

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `KECCAK_RND-C11.i` | x ∈ [0, 4], y ∈ [0, 4], z ∈ [0, 7] | `BYTE_ALU[chi_ANDs[x][y][z]; ⧼AND⧽, 255 - pi[(x + 1) mod 5][y][z], pi[(x + 2) mod 5][y][z]]` | μ |
| `KECCAK_RND-C12.i` | x ∈ [0, 4], y ∈ [0, 4], z ∈ [0, 7] | `BYTE_ALU[chi[x][y][z]; ⧼XOR⧽, pi[x][y][z], chi_ANDs[x][y][z]]` | μ |

Lastly, the round constants are added to one of the lanes in the state. `iota` contains the updated lane. In the definition of `out`, the output of `chi` and `iota` is combined to construct the output of the permutation.

| Tag | Range | Description | Multiplicity |
|-----|-------|-------------|--------------|
| `KECCAK_RND-C13.i` | z ∈ [0, 7] | `BYTE_ALU[iota[z]; ⧼XOR⧽, chi[0][0][z], rc[z]]` | μ |

Lastly, the round chip contributes the following interactions to the lookup:

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `KECCAK_RND-C14` | `KECCAK[timestamp, round, start]` | -μ |
| `KECCAK_RND-C15` | `KECCAK[timestamp, round + 1, out]` | μ |
| `KECCAK_RND-C16` | `KECCAK_RC[rc; round]` | -μ |

### Notes/potential optimizations

- one does not have to repeat `addr` in `state_ptr`; this saves 4 columns and 4 `IS_HALF` checks. - step `rho` does not need to be applied to `state[0][0]`; its has a zero-shift. This saves 16 columns and 4 `HWSL` interactions. - when the output of `HWSL` are `Byte`s mapped as `Half`s, we find that out of every four output bytes, at least one is zero. Since `rnc` is constant, [keccak:c:rho_rotation] makes those zero-bytes show up in `rot_left` and `rot_right` at constant locations. This means 96 columns can be removed from the chip at no cost. Likewise, 96 `IS_BYTE` interactions can be dropped from [keccak:c:range_rot_left] and [keccak:c:range_rot_right]. - the shift-constants are equivalent to `1 mod 16` for `(`x`, `y`) = (1, 0)` and `-1 mod 16` for `(2, 3)`. This means that for those lanes it suffices to constrain `rot_left`/`rot_right` as `Bit`s rather than `Byte`s, saving an additional 8 `IS_BYTE` interactions. - ``rc[2]` = `rc[4]` = `rc[5]` = `rc[6]` = 0`. As such, those elements need not be stored in `rc`, and need not be XORed into the state in the `iota`-step. This saves 8 columns and 4 `XOR_BYTE` interactions. - when executed in large volumnes, `KECCAK_RND` could benefit from having a three-way XOR lookup table. With this in place, the 80 interactions in [keccak:c:theta_cxz_start] and [keccak:c:theta_cxz] could be dropped. Likewise, 80 columns could be removed from the chip (a \~5% savings).

## Round constant lookup

### Columns

We provide the round constants through a short precomputed lookup table: .

### Input

| Name | Type | Description |
|------|------|-------------|
| `round` | `BaseField` |  |
| `RC` | `Byte[8]` | round constants for the given `round` |

### Multiplicity

| Name | Type | Description |
|------|------|-------------|
| `μ` | `BaseField` |  |

| Tag | Description | Multiplicity |
|-----|-------------|--------------|
| `KECCAK_RC-C1` | `KECCAK_RC[RC; round]` | -μ |