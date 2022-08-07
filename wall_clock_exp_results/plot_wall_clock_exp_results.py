import os
import json
from datetime import timedelta
from dateutil import parser


RESULT_FILE_NAME = 'metrics.json'
WIN_RATE_KEY = {
    'smac_lite': 'test_win_rate_mean',
    'smac_full': 'test_battle_won_mean'
}
NUM_TRIALS = 5


def _get_elapsed_timedelta(start_time, end_time):
    _start_time = parser.parse(start_time)
    _end_time = parser.parse(end_time)
    return _end_time - _start_time


def _get_stats_dict(_dir, _algo, win_rate_key):
    _dir = os.path.join(_dir, _algo)
    _time_deltas = []

    for trial in range(1, NUM_TRIALS + 1):
        result_json_path = os.path.join(_dir, str(trial), RESULT_FILE_NAME)
        with open(result_json_path, 'r') as f:
            _json = json.load(f)
            start_time = _json[win_rate_key]['timestamps'][0]
            end_time = _json[win_rate_key]['timestamps'][-1]
            _time_deltas.append(_get_elapsed_timedelta(start_time, end_time))

    mean_time_delta = sum(_time_deltas, timedelta()) / len(_time_deltas)
    median_time_delta = sorted(_time_deltas)[len(_time_deltas)//2]
    max_time_delta = max(_time_deltas)
    min_time_delta = min(_time_deltas)
    return {
        'mean': str(mean_time_delta),
        'median': str(median_time_delta),
        'max': str(max_time_delta),
        'min': str(min_time_delta)
    }


if __name__ == '__main__':
    for dir in ('smac_lite', 'smac_full'):
        for algo in ('qmix', 'iql'):
            print(f'Time stats for {dir}[{algo}]')
            print(json.dumps(_get_stats_dict(dir, algo, WIN_RATE_KEY[dir])))
        print('#'*120)

