import numpy as np
import multiprocessing as mp


def generate_seed(seed):
    seed_seq = np.random.SeedSequence(seed)
    return seed_seq.entropy


if __name__ == "__main__":
    num_seeds = 10
    pool = mp.Pool()
    seeds = pool.map(generate_seed, range(num_seeds))
    print(seeds)
