import fastrand
import matplotlib.pyplot as plt

LOG_L = 16
N = 16

L = 2**LOG_L
LOG_S = LOG_L * N
S = L**N

P = 2**256 - 2**32 - 977
assert P < S


def limbify(x: int) -> list[int]:
    assert x < S

    limbs = []
    for _ in range(2*N):
        (x, limb) = divmod(x, L)
        limbs.append(limb)
    return limbs

def limb_prod(x: list[int], y: list[int]) -> list[int]:
    return [
        sum(x[j] * y[i-j] for j in range(i+1))
        for i in range(2*N) 
    ]
    

def delimbify(limbs: list[int]) -> int:
    return sum(y * pow(L, i) for i, y in enumerate(limbs))

limbs = list[int]
pair = tuple[limbs, limbs]
pairs = tuple[list[pair], list[pair]]

def carries_from_sum_of_products(inp: tuple[list[int], list[int]]) -> int:
    """Given pairs of integers `(a, b)`, compute the maximum carry encountered 
    when computing the sum of the products of their limb decompositions."""
    raw_prods = [limb_prod(*pair) for pair in inp]
    raw_prod_limbs = [sum(elts) for elts in zip(*raw_prods)]    
    
    carry, carries = 0, []
    for limb in raw_prod_limbs:
        carry = (limb + carry) >> LOG_L
        carries.append(carry)
    return carries

def experiment(iters: int, nr_pairs: int, plot_individual: bool):
    carries_per_limb = [[] for _ in range(2*N)]
    for _ in range(iters):
        pairs = [
            (
                [fastrand.pcg32bounded(L) for _ in range(N)] + [0] * N,
                [fastrand.pcg32bounded(L) for _ in range(N)] + [0] * N
            )
            for _ in range(nr_pairs)
        ]
        
        carries = carries_from_sum_of_products(pairs)
        for limb_idx in range(2*N):
            carries_per_limb[limb_idx].append(carries[limb_idx])

    carries_per_limb = [sorted(carries) for carries in carries_per_limb]
    
    if plot_individual:
        # individual plots
        for limb_idx in range(2 * N):
            plt.hist(carries_per_limb[limb_idx], bins=100)
            plt.vlines(2**20, 0, 1000, color='0')
            ax = plt.gca()
            ax.set_xlabel("carry value")
            ax.set_ylabel("frequency")
            ax.set_title(f"Frequency carry c_{limb_idx} ({LOG_L=}, {N=}, μ={nr_pairs})")
            plt.savefig(f"tooling/figures/max_carries/{iters=}_{nr_pairs=}_{limb_idx=}.png")
            plt.clf()
    
    # combined plot
    plt.hist([a for b in carries_per_limb for a in b], bins=100)
    plt.vlines(2**20, 0, 30000, color='0')
    ax = plt.gca()
    ax.set_xlabel("carry value")
    ax.set_ylabel("frequency")
    ax.set_title(f"Combined frequency of all carries ({LOG_L=}, {N=}, μ={nr_pairs})")
    plt.savefig(f"tooling/figures/max_carries/{iters=}_{nr_pairs=}_combined.png")
    plt.clf()

if __name__ == "__main__":
    for i in range(1, 5):
        experiment(50_000, i, False)