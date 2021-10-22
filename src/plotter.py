import matplotlib.pyplot as plt
import matplotlib.tri as tri
import pandas as pd
import numpy as np

def plot_phasediagram(results_file,eci,style='ggplot'):

    results = pd.read_csv(results_file)

    plt.style.use(style)
    for T in results['T'].unique():
        plt.plot(results[results['T'] == T]['1-point_corr'],
                 results[results['T'] == T]['F'],
                 label = f'T = {T}'
                )

    plt.xlim(-1,1)
    plt.title(f"ECI : {eci[2]} :: {eci[3]}")
    plt.legend()
    plt.ylabel('Temperature -->')
    plt.xlabel('1-point Correlation -->')
    plt.savefig(f'pd.png',dpi=300)

    plt.show()

def plot_NN(results_file,eci,style='ggplot'):

    results = pd.read_csv(results_file)

    plt.style.use(style)
    plt.plot(results[results['1-point_corr'] == 0.0]['T'],
             results[results['1-point_corr'] == 0.0]['corrs'].str[2],
            )
    plt.xlim(results['T'].min(),results['T'].max())
    plt.title(f"ECI : {eci[2]} :: {eci[3]}")
    plt.ylabel('NN Correlation -->')
    plt.xlabel('Temperature -->')
    plt.savefig(f'2NN.png',dpi=300)

    plt.show()

def plot_NN_triangle(results_file,eci,single_temp,single=True,gif=False,style='ggplot'):

    results = pd.read_csv(results_file)
    temperatures = results['T'].unique()

    plt.style.use(style)
    if not single:
        fig, axs = plt.subplots(nrows = ceil(len(temperatures)/2),
                                ncols = floor(len(temperatures)/2),
                                figsize=(28, 28),
                                dpi=300)
    else:
        fig, axs = plt.subplots(nrows=1,figsize=(8, 6),dpi=300)

    for i, ax2 in enumerate(fig.axes):

        if not single:
            try:
                temp = temperatures[i]
            except IndexError:
                continue
        else:
            temp = single_temp

        Xs = results[results['T'] == temp]['1-point_corr'].values
        Ys = results[results['T'] == temp]['2-point_corr'].values
        Zs = results[results['T'] == temp]['F'].values

        ax2.set_title(f'T = {temp}')

        ax2.tricontour(Xs, Ys, Zs, levels=10, linewidths=0.5,)
        cntr2 = ax2.tricontourf(Xs, Ys, Zs, levels=14,)

        fig.colorbar(cntr2, ax=ax2,label='F',)
        ax2.plot(Xs, Ys, 'ko', ms=3)
        ax2.set(xlim=(-1, 1), ylim=(-1, 1))

        plt.xlabel("1-point Correlation")
        plt.ylabel("2-point NN Correlation")
        plt.subplots_adjust(hspace=0.5)

    plt.tight_layout()
    plt.savefig()

