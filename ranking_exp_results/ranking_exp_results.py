import os
import json

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

RESULT_FILE_NAME = 'metrics.json'
WIN_RATE_KEY = {
    'lite': 'test_win_rate_mean',
    'full': 'test_battle_won_mean'
}
NUM_TRIALS = 10
INDEX = None
INDEX_PARSED = False


def _get_df_from_dir(_dir, _algo, win_rate_key):
    global INDEX_PARSED
    global INDEX
    _dir = os.path.join(_dir, _algo)
    _results_dict = {str(trial): 1 for trial in range(1, NUM_TRIALS + 1)}

    for trial in range(1, NUM_TRIALS + 1):
        result_json_path = os.path.join(_dir, str(trial), RESULT_FILE_NAME)
        with open(result_json_path, 'r') as f:
            _json = json.load(f)
            _results_dict[str(trial)] = _json[win_rate_key]['values']
            if not INDEX_PARSED:
                INDEX = _json[win_rate_key]['steps']
                INDEX_PARSED = True

    _df = pd.DataFrame.from_dict(_results_dict)
    _df['mean'] = _df.mean(axis=1)
    _df['median'] = _df.median(axis=1)

    return _df[['mean', 'median']]


def plot_graph(_smac_subdir, win_rate_key, title, output_file_name=None):
    iql_df = _get_df_from_dir(_smac_subdir, 'iql', win_rate_key)
    qmix_df = _get_df_from_dir(_smac_subdir, 'qmix', win_rate_key)
    iql_df['qmix_median'] = qmix_df['median']
    combined_df = iql_df[['median', 'qmix_median']]
    combined_df['median'] = combined_df['median']*100
    combined_df['qmix_median'] = combined_df['qmix_median']*100

    combined_df.index = [t/1e6 for t in INDEX]
    combined_df = combined_df.rename(columns={
        'median': 'IQL',
        'qmix_median': 'QMIX'
    })

    sns.set_style('whitegrid')
    ax = sns.lineplot(data=combined_df, palette="tab10", linewidth=2.5)
    ax.set(title=title, xlabel='Timestep (Million)', ylabel='Median Test Win Rate %')
    plt.tight_layout()

    if output_file_name is None:
        plt.show()
    else:
        plt.savefig(output_file_name)
    plt.clf()


if __name__ == '__main__':
    plot_graph('smac_lite', WIN_RATE_KEY['lite'], 'SMAC-lite', 'exp_1_lite.png')
    plot_graph('smac_full', WIN_RATE_KEY['full'], '5m_vs_6m (SMAC)', 'exp_1_full.png')
