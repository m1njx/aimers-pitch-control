"""EXP15: push the small-tree direction (structural regularisation = built-in slope shrinkage)."""
import sys
sys.path.insert(0, '~/LG_data/scratch')
from agent3_exp14_leaves_fat import go

if __name__ == '__main__':
    for lv in [3, 4, 6, 8, 10, 12]:
        go(f'lv{lv} n250', ['sit'], (7, 123), dict(num_leaves=lv))
    print()
    for lv, n, lr in [(8, 800, 0.02), (10, 800, 0.02), (10, 1500, 0.01), (6, 1500, 0.01), (15, 1500, 0.01)]:
        go(f'lv{lv} n{n} lr{lr}', ['sit'], (7, 123), dict(num_leaves=lv, n_estimators=n, learning_rate=lr))
    print()
    for lv in [8, 10]:
        go(f'lv{lv} +fat', ['sit', 'fat'], (7, 123), dict(num_leaves=lv))
        go(f'lv{lv} nosit', [], (7, 123), dict(num_leaves=lv))
