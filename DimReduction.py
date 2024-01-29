import os.path
import umap
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import Frequencies

"""

Running command should be:
python3 DimReduction.py -ind indicator -t threshold_value -p fastas_directory_path

For example:
python3 DimReduction.py -ind E-value -t 1e-10

OR:

python3 DimReduction.py


IMPORTANT: Be sure to include Resfam-full.hmm, the resfam metadata excel file and all desired fasta files in the same 
directory, along with Colors.json, MergeByID.json, MergeByAccession.json and UMAPColors.json! 

DimReduction: This code can receive an indicator, a threshold and a save indicator as an input. It can also receive 
no input. If input was inserted, The code runs the main() function from Frequencies.py module. The code then uses UMAP
technique to reduce dimensions, and plots the result with samples from the same environment having the same color. 
Finally, the code saves the plots as .png files.  

"""


def main(arguments):
    """This function receives input from CMD. Then, it runs the function based on input inserted:
    No input inserted - The function runs umap_from_excel().
    Input inserted - The function runs umap_from_scratch()."""
    input_values = [arguments.indicator, arguments.threshold, arguments.path]
    if input_values.count(None) == len(input_values):
        return umap_from_excel()
    return umap_from_scratch(arguments)


def umap_from_excel():
    """This function runs in case no input was given when running the module.
    The function assumes a "Frequencies" directory exists with all relevant Excel files output by Frequencies.py module.
    The function then runs, for every excel table, the dim_red() function."""
    directory = f"{os.getcwd()}/Frequencies"
    env_df = pd.read_csv(f"{directory}/Frequencies_of_All_Train_Environments.csv", index_col=0)
    env_cols = Frequencies.receive_json(os.path.abspath("Columns of Environments by Criteria.json"))
    for crit in env_cols.keys():
        table = env_df[env_cols[crit]]
        dim_red(table, crit)


def umap_from_scratch(inputs):
    """This function runs in case an input was given when running the module.
    The function runs the Frequencies.py module.
    Then, for every table, the function runs the dim_red() function."""
    env_df, env_cols = Frequencies.main(inputs)[2:4]
    for crit in env_cols.keys():
        table = env_df[env_cols[crit]]
        dim_red(table, crit)


def dim_red(freqs_table, criterion):
    """This function uses UMAP technique to reduce dimensions of the frequencies tables output from Frequencies.py module.
    The function plots the projections as a png file, to a directory.
    The function has no return value.
    Thanks to ChatGPT for assisting with an issue in the plots' legend."""
    freqs_table = freqs_table.reset_index()
    reducer = umap.UMAP(n_components=2, n_neighbors=3, min_dist=0.005, metric='euclidean')
    scaled_data = StandardScaler().fit_transform(freqs_table[freqs_table.columns[1:]])
    embedding = reducer.fit_transform(scaled_data)
    samples = list(freqs_table[freqs_table.columns[0]])
    colors = Frequencies.receive_json(os.path.abspath("Colors.json"))
    color_mapping = {sample: colors[i] for i, sample in enumerate(samples)}
    fig, ax = plt.subplots()
    handles = []
    labels = []
    for i in range(embedding.shape[0]):
        label = samples[i]
        if label not in labels:
            scatter = ax.scatter(embedding[i][0], embedding[i][1], s=75, color=color_mapping[label])
            handles.append(scatter)
            labels.append(label)
        else:
            # Update the color of the existing scatter plot object for the same label
            scatter = ax.scatter(embedding[i][0], embedding[i][1], s=75, color=color_mapping[label])
    ax.legend(handles, labels, title='Environments', loc=(1.1, 0), ncol=int(np.ceil(len(labels) / 20)) + 1)
    plt.gca().set_aspect('equal', 'datalim')
    title = f"UMAP Projection of {criterion}"
    plt.title(title, fontsize=16)
    path = os.path.join(os.getcwd(), 'UMAP_Projections')
    try:
        os.mkdir(path)
    except OSError:
        pass
    plt.savefig(os.path.join(path, f"{title}.png"), bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--indicator', '-ind', type=str, help='Insert sort indicator. Can be E-value or Score only',
                        required=True)
    parser.add_argument('--threshold', '-t', type=float, help='Insert threshold', required=True)
    parser.add_argument('--path', '-p', type=str, help='Insert path for fasta files. The code takes all files that '
                                                       'end with "contigs.min10k.proteins.faa".', required=False)
    args = parser.parse_args()
    main(args)
