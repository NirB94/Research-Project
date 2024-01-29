import argparse
import Frequencies
import DimReduction
import RandomForestModel

"""
Running command should be:
python3 PreprocessingModeling.py -ind indicator -t threshold_value

For example:
python3 PreprocessingModeling.py -ind E-value -t 1e-10
"""


def main(arguments):
    """This function receives arguments from the CMD, validates the input and then starts the process.
    The function runs the whole process from data preprocessing to model evaluation.
    For further details please see relevant modules."""
    indicator = arguments.indicator
    assert (indicator == 'E-value' or indicator == 'Score'), 'Please enter E-value or Score in that exact way.'
    threshold = arguments.threshold
    k = arguments.k if arguments.k is not None else 5
    excel_file = '180102_resfams_metadata_updated_v1.2.2_with_CARD_v2.xlsx'
    Frequencies.plot_all(indicator, threshold, excel_file)
    DimReduction.umap_from_excel()
    RandomForestModel.run_all(k)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--indicator', '-ind', type=str, help='Insert sort indicator. Can be E-value or Score only',
                        required=True)
    parser.add_argument('--threshold', '-t', type=float, help='Insert threshold', required=True)
    parser.add_argument('--k', '-k', type=int, help='Insert number of desired folds for the Cross Validation training. '
                                                    'Default is 5', required=False)
    parser.add_argument('--path', '-p', type=str, help='Insert path for fasta files. The code takes all files that '
                                                       'end with "contigs.min10k.proteins.faa".', required=False)
    args = parser.parse_args()
    main(args)
