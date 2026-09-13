import numpy as np
import random
def random_gen(bit_length, min_val = 0, max_val = 1):
    gen_genome = [str(np.random.randint(min_val, max_val)) for _ in range(bit_length)]
    # gen_genome = [str(random.randint(min_val, max_val)) for _ in range(bit_length)]
    return gen_genome

print(random_gen(21))