#! python3
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

def plot_perf(cpu_time, relative_obj_vals, rel_grad_obj_vals, title, filename, 
              algorithm: str = "Steepest Descent", model: str = None):
    """Plots the data"""
    
    x = 'Iteration, r'
    y1 = 'CPU time (sec)'
    y2 = r'Relative objective value, $f(x^k) - f(x^*)$'
    y3 = r'$\dfrac{\Vert \nabla f(x^k)\Vert}{\Vert \nabla f(x^0) \Vert}$'
    plots_num = 3 if rel_grad_obj_vals else 2
    
    f = plt.figure(figsize=(12, 9), dpi=120)
    
    ax1 = plt.subplot(plots_num, 1, 1)
    ax1.plot(cpu_time, label='CPU time (sec)')  #, list(range(len(cpu_time))))
    ax1.set_xlabel(x)
    ax1.set_ylabel(y1)
    ax1.legend(loc=2)
    
    ax2 = ax1.twinx()
    ax2.plot(np.cumsum(cpu_time), c='y', label='Cumulative CPU time')
    ax2.set_ylabel('Cumulative CPU time')
    ax2.legend(loc=1)

    # plt.plot()
##    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.title(f'{y1} vs {x}')

    plt.subplot(plots_num, 1, 2) 
    plt.plot(relative_obj_vals, 'red')
    plt.xlabel(x)
    plt.ylabel(y2)
    plt.title(f'{y2} vs {x}')

    if rel_grad_obj_vals:
        plt.subplot(plots_num, 1, 3) 
        plt.plot(rel_grad_obj_vals, 'green')
        plt.xlabel(x)
        plt.ylabel(y3)
        plt.title(f'{y3} vs {x}')
    
    plt.suptitle(f'Model: {model}   Algorithm: {algorithm} ({title})')
    plt.tight_layout()

    ax = f.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    os.makedirs('figs', exist_ok=True)
    plt.savefig(fr'figs\{filename}.png')
    
    # plt.close()
