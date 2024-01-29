import os
import numpy as np
from collections import defaultdict
from Bio import SearchIO
from Bio import SeqIO
import pandas as pd
import argparse

"""

Running the function should be:
python3 parseTblout.py -fasta fasta_name -ind indicator -t threshold_value -excel excel_file
    
    
IMPORTANT:
Be sure to include Resfam-full.hmm, the resfam metadata excel file and the fasta file in the same directory!

Parse Tblout
This code receives a fasta format file (*.fasta), a threshold (float or int), and a string indicating whether we want
to filter out data by E-value or by score.
The code runs HMM on the fasta file, and creates a tblout file.
The code parses the table into a chart containing the following:
ID of gene searched, Query name, Query accession, Best score, Best E-value, and a boolean value indicating whether the
protein passed the input threshold or not.
"""


def main(arguments):
    """This function receives arguments from the CMD, validates the input and then starts the process."""
    fasta_file = arguments.fasta_file
    indicator = arguments.indicator
    assert (indicator == 'E-value' or indicator == 'Score'), 'Please enter E-value or Score in that exact way.'
    threshold = arguments.threshold
    excel_file = arguments.excel_file
    return parse(fasta_file, indicator, threshold, excel_file)


def read_fasta_id(fasta_path):
    """This function receive a path to a fasta file.
    The function returns a list of all the ID's in that file."""
    fasta_handle = SeqIO.parse(fasta_path, 'fasta')
    fasta_id = []
    for seq_record in fasta_handle:
        fasta_id.append(seq_record.id)
    return fasta_id


def get_missing_ids(original, new):
    """This function receives a list of original ID's from a fasta file, and a list of ID's of the HMMER output.
    The function returns a  list of all ID's that were omitted during HMMER process."""
    missing = []
    for ID in original:
        if ID in new:
            new.remove(ID)
        else:
            missing.append(ID)
    return missing


def receive_path(path):
    """This function receives a path to a tblout file.
    The function reads the file and converts it to a DataFrame."""
    attribs = ['id', 'query_id', 'accession', 'evalue', 'bitscore']
    hits = defaultdict(list)
    try:
        with open(path) as handle:
            for queryresult in SearchIO.parse(handle, 'hmmer3-tab'):
                for hit in queryresult.hits:
                    for attrib in attribs:
                        if attrib == 'accession':
                            hits[attrib].append(queryresult.accession)
                        else:
                            hits[attrib].append(getattr(hit, attrib))
    except FileNotFoundError:
        return pd.DataFrame()
    return pd.DataFrame.from_dict(hits).rename(columns={'id': 'ID', 'query_id': 'Query_ID',
                                                        'accession': 'Query_Accession', 'evalue': 'E-value',
                                                        'bitscore': 'Score'})


def parse_by_id(table):
    """This function receives a DataFrame based on HMMER output.
    The function parses the table to a list of tables.
    Each list is a DataFrame of each ID."""
    id_set = set(table['ID'])
    tables_list = []
    for ID in id_set:
        tables_list.append(table.loc[table['ID'] == ID])
    # tables_list = np.asarray(tables_list, dtype=object)
    return tables_list


def sort_tables_and_reparse(tables, indicator, thresh):
    """The function receives a list of DataFrames, an indicator and a threshold.
    The function sorts each ID by the indicator.
    The function then merges back all the DataFrames into a single DataFrame.
    The function then adds a column indicating whether the indicator value of each row has passed the threshold.
    The function returns this DataFrame."""
    # Sorts each ID by indicator, and then parses back with a Passed threshold column
    for table in tables:
        table = table.sort_values(by=[indicator], ascending=(indicator != 'Score'))
    new_table = pd.DataFrame([table.iloc[0] for table in tables])
    booleans = new_table[indicator] > thresh if indicator == 'Score' else new_table[indicator] < thresh
    new_table['Passed threshold'] = booleans
    return new_table


def parse(fasta_file, indicator, thresh, excel_path):
    """This function receives a fasta file, an indicator, a threshold and an excel file.
    The function load HMMER module.
    The function then runs the HMMER process on the fasta file, and saves it to a tblout file.
    The function converts the tblout file to a DataFrame.
    The function then merges the excel file to the DataFrame by Query Accession - gene families.
    The function then splits the DataFrame by ID and merges back the table by best values of each ID, according to the
    indicator, into a single DataFrame.
    The function then adds missing ID's that were omitted during HMMER process.
    the function exports the DataFrame into an excel (xlsx) file.
    The function returns the DataFrame."""
    os.system("module load hmmer")
    fasta_name = os.path.basename(fasta_file)
    fasta_name = str(os.path.splitext(fasta_name)[0])
    tblout_name = f"{fasta_name}.tblout"
    excel = pd.read_excel(excel_path)
    os.system(f"hmmsearch -o /dev/null --tblout {tblout_name} Resfams-full.hmm {fasta_file}")
    table = receive_path(tblout_name)
    if len(table.index) == 0:
        return table
    table = pd.merge(table, excel, left_on="Query_Accession", right_on="ResfamID", how='left')
    all_ids = read_fasta_id(fasta_file)
    table_ids = table['ID'].values.tolist()
    missing_ids = get_missing_ids(all_ids, table_ids)
    missing_ids = pd.DataFrame(missing_ids, columns=['ID'])
    tables_list = parse_by_id(table)
    new_table = sort_tables_and_reparse(tables_list, indicator, thresh)
    os.system(f"rm {tblout_name}")
    final_table = pd.concat([new_table, missing_ids], ignore_index=True)
    return final_table


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fasta_file', '-fasta', type=str, help='Insert fasta file name. Must end with .faa or .fasta',
                        required=True)
    parser.add_argument('--indicator', '-ind', type=str, help='Insert sort indicator. Can be E-value or Score only',
                        required=True)
    parser.add_argument('--threshold', '-t', type=float, help='Insert threshold', required=True)
    parser.add_argument('--excel_file', '-excel', type=str, help='Insert excel file name.', required=True)
    args = parser.parse_args()
    main(args)
