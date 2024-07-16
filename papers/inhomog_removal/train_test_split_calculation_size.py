"""
Im a little bit stupid

So I made a manual calculation of the train/test/validation size...
"""

n_b1 = 23
n_pat = 40
p_train = 0.7
p_test = 0.2
p_val = 0.1
n_slice = 90
pat_total = 0
b1_total = 0
for p_sel in [p_test, p_val]:
    n_pat_sel = int(n_pat * p_sel)
    n_b1_sel = int(n_b1 * p_sel)
    print(n_pat_sel * n_b1_sel * n_slice)
    print(f'\t {n_pat_sel} - {n_b1_sel}')
    pat_total += n_pat_sel
    b1_total += n_b1_sel

print((n_pat - pat_total) * (n_b1 - b1_total) * n_slice)

nom = (p_train * n_pat * p_train * n_b1)
denom = (p_train * n_pat * p_train * n_b1 + p_test * n_pat * p_test * n_b1 + p_val * n_pat * p_val * n_b1)
nom/denom