
"""
**small excursion into the sets of unqiue sums
"""


import itertools
import numpy as np

for _ in range(10):
    x0 = np.random.randint(0, 10)
    x1 = np.random.randint(0, 10)
    print(x0, x1)
    derp = [x0, x1]
    n_max = 25
    n_iter = 0
    while n_iter < n_max:

        possible_combinations = [x[0] + x[1] for x in itertools.product(derp, derp)]
        total_comb = set(possible_combinations).union(set(derp))
        min_total = min(total_comb)
        max_total = max(total_comb)

        diff_set = set(range(min_total, max_total +1)).difference(total_comb)

        if diff_set:
            print('APpending ', min(diff_set))
            derp.append(min(diff_set))
        else:
            print('APpending ', max_total +1)
            derp.append(max_total +1)

        n_iter += 1

    import matplotlib.pyplot as plt

    # plt.plot(range(0, max(derp)), derp)
    # plt.plot(derp)
    # plt.plot(np.exp(np.arange(0,10)), 'k')
