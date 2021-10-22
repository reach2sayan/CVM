import matplotlib.pyplot as plt
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

def plot
