import glob
import matplotlib.pyplot as plt
import numpy as np
import opt
import os
import sys

def plot_batch(files, names, colors, markers, style, score='score', fix=True, markevery=1):
    npzs = [np.load(f) for f in files]
    results = [[f[name] for f in npzs] for name in names]
    for f in npzs: f.close()

    for name, color, marker, result in zip(names, colors, markers, results):
        c, s = opt.averages(result, score, fix)
        plt.plot(c, s, color=color, linewidth=2, linestyle=style,
                 marker=marker, markerfacecolor='none', markersize=7.0, markeredgewidth=1.5,
                 markeredgecolor=color, markevery=markevery)

def plot_averages(result_dir, costs, names, colors, markers, do_uniform=True, score='score'):
    speed_files = glob.glob('%s/cost.*.npz' % result_dir)
    speed_results = [[np.load(f)[name] for f in speed_files] for name in names]

    if do_uniform:
        uniform = np.load('%s/uniform.npz' % result_dir)
        regular_results = [[opt.recompute_costs(uniform[name], c) for c in costs] for name in names]

    for name, color, marker, result in zip(names, colors, markers, speed_results):
        c, s = opt.averages(result, score)
        plt.plot(c, s, color=color, linewidth=2,
                 marker=marker, markerfacecolor='none', markersize=7.0, markeredgewidth=1.5,
                 markeredgecolor=color, markevery=100)

    if do_uniform:
        for name, color, result in zip(names, colors, regular_results):
            c, s = opt.averages(result, score)
            plt.plot(c, s, color=color, linewidth=2, linestyle='--',
                 marker=marker, markerfacecolor='none', markersize=7.0, markeredgewidth=1.5,
                 markeredgecolor=color)

def plot_single(results, names, colors, markers, style, score='score'):
    named_results = [results[n] for n in names]
    for name, color, marker, result in zip(names, colors, markers, named_results):
        c, s = opt.fix(result['cost'], result[score])
        plt.plot(c, s, color=color, linewidth=2, linestyle=style,
                 marker=marker, markerfacecolor='none', markersize=7.0, markeredgewidth=1.5,
                 markeredgecolor=color)

def plot_groups(files, score='score', fix=True, markevery=1):
    plot_batch(files, ['OPT', 'FR', 'OMP'], 'krgb', 's+o^', '-', score=score, fix=fix, markevery=markevery)
    plot_batch(files, ['OMP NOINV'], 'g', 'o', ':', score=score, fix=fix, markevery=markevery)
    plot_batch(files, ['FR SINGLE', 'OMP SINGLE'], 'rgb', '+o^', '--', score=score, fix=fix, markevery=markevery)

    plt.legend(('OPT', 'FR - Grouped', 'OMP - Grouped', 'OMP - Grouped Naive', 'FR - Single', 'OMP - Single'),
               loc='lower right', prop={'size': 18})

if __name__ == '__main__':
    names = ['FR', 'OMP', 'L1']
    colors = ['r', 'g', 'b']

    plt.figure()
    result_dir = sys.argv[1]
    costs = np.load(sys.argv[2])['all_costs']

    plot_averages(result_dir, costs, names, colors, [None]*3)

    plt.xlabel('Cost')
    plt.ylabel('$R^2$')
    plt.show()
