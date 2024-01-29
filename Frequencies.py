import json
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import parseTblout as ptb
import argparse
from pathlib import Path

"""

Running command should be:
python3 Frequencies.py -ind indicator -t threshold_value

For example:
python3 Frequencies.py -ind E-value -t 1e-10

IMPORTANT: Be sure to include Resfam-full.hmm, the resfam metadata excel file, parseTblout.py and DimReduction.py 
modules in the same directory as this, along with Colors.json, MergeByID.json and MergeByAccession.json! 

Frequencies: This code receives an indicator, and a threshold as an input. The code runs the parse() 
function from parseTblout module for every fasta file in /davidb/assemblies/denovo path, with the excel file and the 
Resfam-full.hmm file, which should also be in the directory. 
The code then builds the frequencies of the queries ID and their families for each sample, and plots it. 
The return value is the frequency tables, and it also creates a png file. 

The code also contains a distance function, which has no use as of this moment.

"""

TRAIN_ENVIRONMENTS = {'Animal Microbiome': [], 'Human Microbiome': [], 'Sewage': [], 'GroundWater': [], 'Soil': []}

TRAIN_MGNIFY_ANIMAL_MICROBIOME = ['coral_metagenome', 'cow_dung_metagenome', 'epibiont_metagenome', 'insect_metagenome', 'mouse_gut_metagenome', 'pig_gut_metagenome']
TRAIN_MGNIFY_HUMAN_MICROBIOME = ['human_oral_metagenome', 'human_gut_metagenome', 'human_skin_metagenome', 'human_metagenome']
TRAIN_MGNIFY_SEWAGE = ['wastewater_metagenome', 'activated_sludge_metagenome', 'sludge_metagenome']
TRAIN_MGNIFY_GROUNDWATER = []
TRAIN_MGNIFY_SOIL = ['beach_sand_metagenome', 'compost_metagenome', 'freshwater_sediment_metagenome', 'marine_sediment_metagenome', 'rhizosphere_metagenome', 'sediment_metagenome', 'soil_metagenome']
TRAIN_MGNIFY_ALL_ENVIRONMENTS = {'Animal Microbiome': TRAIN_MGNIFY_ANIMAL_MICROBIOME, 'Human Microbiome': TRAIN_MGNIFY_HUMAN_MICROBIOME,
                         'Sewage': TRAIN_MGNIFY_SEWAGE, 'GroundWater': TRAIN_MGNIFY_GROUNDWATER, 'Soil': TRAIN_MGNIFY_SOIL}

TRAIN_WGS_ANIMAL_MICROBIOME = ['coral_metagenome', 'insect_metagenome', 'fish_metagenome', 'pig_gut_metagenome', 'sponge_metagenome', 'invertebrate_metagenome', 'epibiont_metagenome', 'mollusc_metagenome', 'termite_gut_metagenome', 'wallaby_gut_metagenome', 'marine_plankton_metagenome', 'mosquito_metagenome', 'scorpion_gut_metagenome', 'insect_gut_metagenome', 'mouse_gut_metagenome', 'jellyfish_metagenome', 'bovine_gut_metagenome', 'chicken_gut_metagenome', 'wasp_metagenome']
TRAIN_WGS_HUMAN_MICROBIOME = ['human_gut_metagenome', 'human_saliva_metagenome', 'human_lung_metagenome', 'respiratory_tract_metagenome', 'human_oral_metagenome', 'human_vaginal_metagenome', 'human_skin_metagenome', 'human_metagenome']
TRAIN_WGS_SEWAGE = ['wastewater_metagenome', 'activated_sludge_metagenome', 'sludge_metagenome']
TRAIN_WGS_GROUNDWATER = ['groundwater_metagenome', 'rock_porewater_metagenome']
TRAIN_WGS_SOIL = ['compost_metagenome', 'freshwater_sediment_metagenome', 'permafrost_metagenome', 'alkali_sediment_metagenome', 'beach_sand_metagenome', 'sediment_metagenome', 'cold_seep_metagenome', 'marine_sediment_metagenome', 'soil_metagenome']
TRAIN_WGS_ALL_ENVIRONMENTS = {'Animal Microbiome': TRAIN_WGS_ANIMAL_MICROBIOME, 'Human Microbiome': TRAIN_WGS_HUMAN_MICROBIOME,
                         'Sewage': TRAIN_WGS_SEWAGE, 'GroundWater': TRAIN_WGS_GROUNDWATER, 'Soil': TRAIN_WGS_SOIL}

TEST_ENVIRONMENTS = {'Animal Microbiome': [], 'Human Microbiome': [], 'Sewage': [], 'GroundWater': [], 'Soil': []}
TEST_ANIMAL_MICROBIOME = ['Bird', 'Bovine', 'Fish', 'Insect', 'PigFarm', 'Rumens', 'Sponge']
TEST_HUMAN_MICROBIOME = ['HMP', 'HumanGut.DNA_RNA']
TEST_SEWAGE = ['global_sewage']
TEST_GROUNDWATER = ['Groundwater', 'Rifle']
TEST_SOIL = ['AustralianSoils', 'AustralianSoils.540', 'CalifornianSoils', 'ForestSoils', 'SoilWarming']
TEST_ALL_ENVIRONMENTS = {'Animal Microbiome': TEST_ANIMAL_MICROBIOME, 'Human Microbiome': TEST_HUMAN_MICROBIOME,
                         'Sewage': TEST_SEWAGE, 'GroundWater': TEST_GROUNDWATER, 'Soil': TEST_SOIL}


def main(arguments):
    """This function receives arguments from the CMD, validates the input and then starts the process."""
    indicator = arguments.indicator
    assert (indicator == 'E-value' or indicator == 'Score'), 'Please enter E-value or Score in that exact way.'
    threshold = arguments.threshold
    excel_file = '180102_resfams_metadata_updated_v1.2.2_with_CARD_v2.xlsx'
    return plot_all(indicator, threshold, excel_file)


def get_all_fastas():
    """ This function is used to get all fasta files to read. The function has no input. The function iterates over
    all the desired environments, and for each environment and sub-environment, it gets all files that end with
    .contigs.min10k.proteins.faa. The function then extracts the name of the file and adds the name and full path to
    two different lists. The function the updates the ENVIRONMENTS dictionary - and inserts the file name to its
    corresponding environment name. The function returns the two mentioned lists - first the path list and then the
    names list.
    """
    train_samples_mgnify_path = '/davidb/assemblies/preassembled/Mgnify/MgyAssemblies/contigs.min2500/'
    train_fasta_paths, train_fasta_files = get_fastas_mgnify(train_samples_mgnify_path)
    train_samples_wgs_path = '/davidb/bio_db/NCBI/WGS/Metagenomes/'
    environments = TRAIN_WGS_ALL_ENVIRONMENTS
    for env_class in environments.keys():
        for env in environments[env_class]:
            curr_path = os.path.join(train_samples_wgs_path, env)
            temp_fasta_paths, temp_fasta_files = get_all_fastas_sample_and_clean(curr_path)
            temp_fasta_names = all_names(temp_fasta_files)
            train_fasta_paths.extend(temp_fasta_paths)
            train_fasta_files.extend(temp_fasta_files)
            TRAIN_ENVIRONMENTS[env_class].extend(temp_fasta_names)
    test_samples_path = '/davidb/assemblies/denovo'
    test_fasta_paths = []
    test_fasta_files = []
    environments = TEST_ALL_ENVIRONMENTS
    for env_class in environments.keys():
        for env in environments[env_class]:
            curr_path = os.path.join(test_samples_path, env)
            temp_fasta_paths, temp_fasta_files = get_all_fastas_sample_and_clean(curr_path)
            temp_fasta_names = all_names(temp_fasta_files)
            test_fasta_paths.extend(temp_fasta_paths)
            test_fasta_files.extend(temp_fasta_files)
            TEST_ENVIRONMENTS[env_class].extend(temp_fasta_names)
    return train_fasta_paths, train_fasta_files, test_fasta_paths, test_fasta_files


def get_key(dic, value):
    """This function is used to get a key out of a dictionary by value.
    If value not in the dictionary, the function returns a None value."""
    for key in dic.keys():
        if value in dic[key]:
            return key
    return None


def get_fastas_mgnify(mgnify_path):
    fastas_paths = []
    fasta_files = []
    subdirectories = []
    for subdir, dirs, files in os.walk(mgnify_path):
        subdirectories.append(subdir)
    subdirectories = sorted(subdirectories[1:])
    for subdir in subdirectories:
        temp_paths = list(Path(subdir).rglob('*.contigs.min10k.proteins.faa'))
        fastas_paths.extend(temp_paths)
        fasta_files.extend([os.path.basename(temp_file) for temp_file in temp_paths])
    i = 0
    while i < len(fasta_files):
        name = fasta_files[i]
        key = get_key(TRAIN_MGNIFY_ALL_ENVIRONMENTS, name.split('.')[0])
        if key is None:
            fasta_files.remove(name)
            fastas_paths.pop(i)
        else:
            TRAIN_ENVIRONMENTS[key].append(receive_name(name, mgnify_flag=True))
            i += 1
    return fastas_paths, fasta_files


def get_all_fastas_sample_and_clean(path):
    """This function is used to get all fasta files from a single sub-environment. The function receives a path.
    The function then gets all files that end with .contigs.min10k.proteins.fasta (or .faa).
    The function then iterates over these files and extracts the full name of the file from its full path.
    The function returns two lists - one of the fasta files path's and one of faste files names."""
    curr_path_list = list(Path(path).rglob("*.contigs.min10k.proteins.fasta"))
    curr_path_list.extend(list(Path(path).rglob("*.contigs.min10k.proteins.faa")))
    curr_name_list = []
    for i in range(len(curr_path_list)):
        curr_name_list.append(str(curr_path_list[i]).split('/')[-1])
    return curr_path_list, curr_name_list


def receive_name(full_name, mgnify_flag=False):
    """ This function is used to get a shorter name for each sample, for convenience purposes. The function receives
    a file's full name. The function then splits the name, and takes the first element (environment) and some
    identifying element.
    The function returns a joined string of these two elements.
    """
    split_name = full_name.split('.')
    slice_index = 2
    if split_name[0] == 'Global_sewage' or split_name[0] == 'Australian_soils':
        slice_index = 3
    elif split_name[0][:2] == 'CG':
        realpath = os.path.realpath(full_name).split('/')
        try:
            realpath.remove('')
        except ValueError:
            pass
        split_name = realpath[3:5] + [split_name[0]]
        slice_index = 3
    if mgnify_flag:
        return ' '.join((split_name[0], split_name[2]))
    return ' '.join(split_name[:slice_index])


def all_names(files_list, mgnify_flag=False):
    names_list = []
    for file in files_list:
        names_list.append(receive_name(file, mgnify_flag))
    return names_list


def sort_names():
    """This function sorts the samples given as an input in the following order: 1) Animal Microbiome. 2) Human
    Microbiome. 3) Global Sewage. 4) Groundwater. 5) Soils. Reason for this order is by similarity.
    The function returns the names list in the described order."""
    my_order = ['Animal Microbiome', 'Human Microbiome', 'Sewage', 'GroundWater', 'Soil']
    order = {key: i for i, key in enumerate(my_order)}
    return order


def parse_all(fastas_lists, indicator, thresh, excel_file, train=True):
    """This function is used to get all tblout tables. The function receives a list of fasta files, indicator,
    threshold and an excel file. The function runs the receive_table() function on each fasta file. If the returned
    table is empty, meaning the fasta file is empty, the function removes the file from the fastas list and from the
    ENVIRONMENT dictionary. The function then removes all rows that have not passed the threshold value. The function
    then removes all .tblout files that somehow were not removed. The function returns a dictionary of these outputs,
    as fasta_name : table as key : value."""
    tables_dict = {}
    to_remove = set()
    for i in range(len(fastas_lists[0])):
        fasta_file = fastas_lists[0][i]
        fasta_name = fastas_lists[1][i]
        table = receive_table(fasta_file, indicator, thresh, excel_file)
        if len(table.index) == 0:
            to_remove.add(fasta_name)
        else:
            table = table[table['Passed threshold'] == True]
            if len(table.index) == 0:  # Meaning no row has passed the threshold value.
                to_remove.add(fasta_name)
            else:
                tables_dict[fasta_name] = table
    remove_from_dict(to_remove, dic=TRAIN_ENVIRONMENTS if train else TEST_ENVIRONMENTS)
    clean_remaining_tblouts()
    return tables_dict


def remove_from_dict(remove_list, dic):
    """This function is used to remove empty fasta files from ENVIRONMENTS global dictionary.
    The function receives a list of fasta files to remove.
    The function has no return value."""
    for deleted in remove_list:
        for env in dic.keys():
            if deleted in dic[env]:
                dic[env].remove(deleted)
                break


def receive_table(fasta_file_path, indicator, thresh, excel_file):
    """The function receives a fasta file, indicator, threshold and an excel file, and runs the parse function of the
    parseTblout module.
    The function returns a DataFrame. For further documentation please see the parseTblout module."""
    return ptb.parse(fasta_file_path, indicator, thresh, excel_file)


def clean_remaining_tblouts():
    """This function is used to remove any tblout file remaining after parse() function of the parseTblout module."""
    for file in os.listdir(os.getcwd()):
        if file.endswith(".tblout"):
            os.system(f"rm {file}")


def build_samples_table(tables, train=True):
    """ This function is used to build the samples-features table.
    The function receives a dictionary which is the output from parse_all() function.
    Then, for each table, the function adds a new Sample column containing a shortened name of the sample, and concats it to a new DataFrame.
    The function removes the all columns that are not in to_keep list.
    The function adds an Environments column using add_environments() function.
    The function then modifies the Drug Class (CARD) column to contain only first element.
    The function then sorts the rows by environments, Order mentioned in sort_names() function.
    The function returns the table.
    """
    to_keep = ['Sample', 'ID', 'Query_ID', 'Query_Accession', 'Mechanism Classification', 'Drug Class (CARD)']
    samples_df = pd.DataFrame()
    for key in tables.keys():
        table = tables[key]
        table['Sample'] = receive_name(key)
        samples_df = pd.concat([samples_df, table])
    samples_df = samples_df[to_keep]
    samples_df = add_environments(samples_df, TRAIN_ENVIRONMENTS if train else TEST_ENVIRONMENTS)
    samples_df['Drug Class (CARD)'] = samples_df[samples_df['Drug Class (CARD)'].notnull()]['Drug Class (CARD)'].apply(lambda x: x.split(';')[0])
    samples_df = samples_df.iloc[samples_df['Environments'].map(sort_names()).sort_values().index]
    return samples_df


def add_environments(df, dic):
    """ This function is used to add Environments column.
    The function receives a DataFrame and a dictionary.
    Dictionary is built by environment : samples as keys : values.
    The function builds a new DataFrame. It's columns are ID and Environments.
    The function then returns a merged DataFrame of the input DataFrame and the new one, by ID."""
    dic_2_df = {}
    for i in range(len(df)):
        sample_name = df.iloc[i]['Sample']  # TODO ORIGINAL WAS 'ID' COLUMN. VARIFY NECESSITY OF CHANGE
        dic_2_df[sample_name] = get_key(dic, sample_name.split('.')[0])
    env_df = pd.DataFrame([dic_2_df]).T.reset_index()
    env_df.columns = ['Sample', 'Environments']  # TODO SAME HERE!!!!
    return df.merge(env_df, how='outer', on='Sample')  # TODO 'Sample' COLUMN SEEMS TO WORK JUST FINE LEAVE IT!


def create_json(to_file, name, path=None):
    """ This function is used to create a json file.
    The function receives the desired variable we wish to convert, desired name, and can also receive a destination.
    If no destination was inserted, default is current directory.
    The function creates a json file from the desired variable with the inserted name."""
    if path is None:
        path = os.getcwd()
    with open(os.path.join(path, f'{name}.json'), 'w') as fout:
        json.dump(to_file, fout)


def receive_json(path):
    """This function receives a destination path for a .json file.
    The function returns the content of the file as a variable."""
    with open(path, 'r') as fin:
        d = json.load(fin)
    return d


def merge_id_and_accession(samples_df):
    """ This function is used to add New Query_ID and Query_Accession columns.21
    The function receives the samples table DataFrame output from build_samples_table() function.
    The function merges Query ID's and Query Accessions by similarity and relativity, That is, queries that belong to
    the same family.
    The function loads json files for merges instructions, and merges the queries.
    The function then maps accordingly.
    The function returns the modified DataFrame. """
    id_path = os.path.abspath('MergesByID.json')
    acc_path = os.path.abspath('MergesByAccession.json')
    id_merges_dic = receive_json(id_path)
    acc_merges_dic = receive_json(acc_path)
    merges_dics = {'Query_ID': id_merges_dic, 'Query_Accession': acc_merges_dic}
    for key in merges_dics.keys():
        samples_df[f'New_{key}'] = samples_df[key].map(merges_dics[key])
    order = ['Sample', 'ID', 'Query_ID', 'New_Query_ID', 'Query_Accession', 'New_Query_Accession',
             'Mechanism Classification', 'Drug Class (CARD)', 'Environments']
    samples_df = samples_df[order]
    new_id_dic = {key: f"Total {key} (ID)" for key in samples_df['New_Query_ID'].unique() if not pd.isna(key)}
    samples_df['New_Query_ID'] = samples_df['New_Query_ID'].map(new_id_dic)
    new_acc_dic = {key: f"Total {key} (Accession)" for key in samples_df['New_Query_Accession'].unique() if not pd.isna(key)}
    samples_df['New_Query_Accession'] = samples_df['New_Query_Accession'].map(new_acc_dic)
    return samples_df


def create_table(samples_df, model=True):
    """ This function is used to create the frequencies table.
    The function receives the samples table output from merge_id_and_accession() function, and a model flag.
    If model is true, it means that we want the frequencies of each sample individually, as we want to train the ML model based on that table.
    If model is false, it means we want the frequencies of each environment.
    The function then, for each criterion per groupby index, builds a frequency table, using build_criterion_freqs_table() function.
    Groupby index is determined by model flag.
    The function merges all the tables into a single table.
    The function keeps all relevant columns for each criterion in a dictionary.
    If groupby is by samples, the function adds an Environment column for classifying purposes.
    The function returns the final frequencies table and the columns dictionary."""
    grpby_index = 'Sample' if model else 'Environments'
    criteria = samples_df.columns[2:-1].to_list()
    criteria_cols = {crit: None for crit in criteria}
    final_df = pd.DataFrame()
    for criterion in criteria:
        temp_freq = build_criterion_freqs_table(samples_df, grpby_index, criterion)
        criteria_cols[criterion] = temp_freq.columns.to_list()
        final_df = pd.concat([final_df, temp_freq], axis=1, join='outer')
    if model:
        final_df.insert(0, 'Environments', np.array([samples_df[samples_df['Sample'] == sample]['Environments'].unique()[0] for sample in samples_df['Sample'].unique()]))
    return final_df, criteria_cols


def build_criterion_freqs_table(samples_df, grpby, crit):
    """ This function is used to build a frequency table for each criterion and by a specified index.
    The function receives the samples table, a groupby index and a criterion.
    The function creates a DataFrame of the frequencies of the criterion by the groupby index.
    The function returns that DataFrame."""
    crit_freq = samples_df.groupby(grpby, sort=False)[crit].value_counts(normalize=True).rename('Frequency').reset_index().pivot_table(index=grpby, columns=crit, values='Frequency', sort=False).fillna(0)
    crit_freq.columns = crit_freq.columns.to_list()
    return crit_freq


def plot_table(table, criterion):
    """ This function is used to plot a frequency table.
    The function receives the frequencies table, and a criterion.
    table contains all the frequencies of the input criterion.
    The function plots that table as a stacked bar plot.
    The function saves the plots as .png files."""
    colors = receive_json(os.path.abspath('Colors.json'))
    ttl = f'Frequencies of Environments by {criterion}'
    try:
        table.plot(kind='bar', stacked=True, title=ttl, ylabel='Frequencies', rot=0, color=colors)
    except TypeError as error:
        print(f"\nError in plotting table: {error}\n")
        table.to_csv(f"{os.getcwd()}/Errored_Table.csv")
    plt.legend(loc=(1.04, -0.15), ncol=(table.shape[1] // 16 if table.shape[1] > 40 else 2))
    plt.subplots_adjust(right=max(1.5, table.shape[0] // 10))
    path = os.path.join(os.getcwd(), 'Frequency_Bars')
    try:
        os.mkdir(path)
    except OSError:
        pass
    plt.savefig(os.path.join(path, f"{ttl}.png"), bbox_inches='tight')


def plot_all_freqs(df, crit_cols):
    """ This function is used to plot all the frequency plots. The function receives the final samples-frequencies
    table output from create_table() function and a dictionary containing the columns for each criterion.
    Then, for each criterion, the function plots the table using plot_table() function. """
    for crit in crit_cols.keys():
        table = df[crit_cols[crit]]
        if crit in ['New_Query_ID', 'New_Query_Accession']:
            splt = crit.split('_')[-1]
            table.columns = [re.split(fr'Total | [(]|{splt}[)]', table.columns[i])[1] for i in
                             range(len(table.columns))]
        plot_table(table, crit)


def calculate_distance(vector1, vector2, func=np.linalg.norm):
    """The function receives 2 vectors and a distance function. The function calculates distance according to any
    distance function. Default is l2 norm."""
    return func(vector1 - vector2)


def plot_all(indicator, thresh, excel_file):
    """This function receives an indicator, a threshold, and an excel file. The function gets all the fasta files
    from the labs inventory. The function runs the parseTblout module to receive DataFrames of each sample. The
    function then, using build_samples_table() and merge_id_and_accession() functions, builds a DataFrame containing
    the following columns: Sample, ID, Query_ID, New_Query_ID, Query_Accession, New_Query_Accessionm,
    Mechanism Classification, Drug Class (CARD) and Environments. The function then builds a frequencies' table twice
    - one per sample for ML model, and one per environment for plotting. The function plots the frequencies,
    and saves the plots as png files. The function saves the tables and the columns dictionaries to xlsx and json
    files respectively. The function returns the frequencies DataFrames and the columns. train_charts_samples and
    train_charts_cols are the tables for plotting and will be used for DumReduction.py module. """
    fastas_lists = get_all_fastas()
    train_tables = parse_all(fastas_lists[:2], indicator, thresh, excel_file, train=True)
    train_samples_table = build_samples_table(train_tables, train=True)
    train_samples_table = merge_id_and_accession(train_samples_table)
    train_model_samples, train_model_cols = create_table(train_samples_table, model=True)
    train_charts_samples, train_charts_cols = create_table(train_samples_table, model=False)
    test_tables = parse_all(fastas_lists[2:], indicator, thresh, excel_file, train=False)
    test_samples_table = build_samples_table(test_tables, train=False)
    test_samples_table = merge_id_and_accession(test_samples_table)
    test_model_samples, test_model_cols = create_table(test_samples_table, model=True)
    path = os.path.join(os.getcwd(), 'Frequencies')
    try:
        os.mkdir(path)
    except OSError:
        pass
    train_model_samples.to_csv(f"{path}/Frequencies_of_All_Train_Samples.csv")
    train_charts_samples.to_csv(f"{path}/Frequencies_of_All_Train_Environments.csv")
    create_json(train_model_cols, "Columns of Train Samples by Criteria", path=path)
    create_json(train_charts_cols, "Columns of Train Environments by Criteria", path=path)
    path = os.path.join(os.getcwd(), 'Frequencies')
    test_model_samples.to_csv(f"{path}/Frequencies_of_All_Test_Samples.csv")
    create_json(test_model_cols, "Columns of Test Samples by Criteria", path=path)
    plot_all_freqs(train_charts_samples, train_charts_cols)
    return train_model_samples, train_model_cols, train_charts_samples, train_charts_cols, test_model_samples, test_model_cols


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--indicator', '-ind', type=str, help='Insert sort indicator. Can be E-value or Score only',
                        required=True)
    parser.add_argument('--threshold', '-t', type=float, help='Insert threshold', required=True)
    args = parser.parse_args()
    main(args)
