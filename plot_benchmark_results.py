import os
import re 
import pandas as pd
import matplotlib.pyplot as plt

def substring_before(s, delim):
    return s.partition(delim)[0]

def substring_after(s, delim):
    return s.partition(delim)[2]

def num_sort(test_string):
    return list(map(int, re.findall(r'\d+', test_string)))[0]

#"results/AMD/AMD_Radeon_780M/AMD_Radeon_780M__half_model_train_benchmark.csv",
#"results/Nvidia/TITAN_RTX/TITAN_RTX__half_model_train_benchmark.csv",
#"results/Nvidia/GeForce_GTX_1080_TI/GeForce_GTX_1080_TI__half_model_training_benchmark.csv",
#"results/Nvidia/GeForce_RTX_2080_TI/GeForce_RTX_2080_TI__half_model_training_benchmark.csv",
#"results/AMD/AMD_Radeon_780M/AMD_Radeon_780M__half_model_train_benchmark.csv",

#                 "results/Nvidia/GeForce_RTX_2060_MaxQ/GeForce_RTX_2060_MaxQ__half_model_train_benchmark.csv",
_result_filename_arr = [
                "results/Nvidia/GeForce_GTX_1080_TI/GeForce_GTX_1080_TI__half_model_training_benchmark.csv",
                "results/AMD/AMD_Radeon_RX_7700S_NO_OPT/AMD_Radeon_RX_7700S_NO_OPT__half_model_train_benchmark.csv",
                "results/AMD/AMD_Radeon_RX_7700S/AMD_Radeon_RX_7700S__half_model_train_benchmark.csv",
                "results/AMD/AMD_Radeon_RX_6800/AMD_Radeon_RX_6800__half_model_train_benchmark.csv",
                "results/Nvidia/GeForce_RTX_3090/GeForce_RTX_3090__half_model_train_benchmark.csv",
                "results/AMD/AMD_Radeon_RX_7900_XTX/AMD_Radeon_RX_7900_XTX__half_model_train_benchmark.csv",
            ]
#                "results/Nvidia/A100_SXM4_40GB/A100_SXM4_40GB__half_model_train_benchmark.csv",

"""
_result_filename_arr = [
                "test1.csv",
                "test2.csv",
            ]
"""
### Filter file name list for files ending with .csv
_result_filename_arr = [file for file in _result_filename_arr if '.csv' in file]

_benchmark_name_arr = ["resnet", "resnext", "densenet", "squeezenet", "vgg", "mobilenet"]
_benchmark_values_dict = {}

_df_means_appended = []
basename_arr = []
### Loop over all files
for result_filename in _result_filename_arr:
    ### Read .csv file and append to list
    df = pd.read_csv(result_filename)

    basename = os.path.basename(result_filename)
    basename = substring_before(basename, "__")
    basename_arr.append(basename)
    
    tittle_str = substring_after(result_filename, "__")
    tittle_str = substring_before(tittle_str, "_benchmark")

    # split benchmarks for each device to multiple result arrays
    # and then collect resutls for multiple gpus to same arrays
    for benchmark_name in _benchmark_name_arr:
        _benchmark_values_dict[benchmark_name] = _benchmark_values_dict.get(benchmark_name, [])
        _df_means_appended = _benchmark_values_dict[benchmark_name]
        cols = [cc for cc in df.columns if cc.startswith(benchmark_name)]
        cols.sort(key=num_sort) 
        df2 = df[cols]
        means = df2.mean()
        _df_means_appended.append(means)

for benchmark_name in _benchmark_name_arr:
    _df_means_appended = _benchmark_values_dict.get(benchmark_name, [])    
    df_mean = pd.concat(_df_means_appended, axis=1)
    df_mean.columns = basename_arr
    df_mean.plot(xlabel='benchmark model', ylabel='execution time (msec), smaller value is better', title=tittle_str)
    # plt.plot(df_mean)
    ### Generate the plot
    plt.show()
