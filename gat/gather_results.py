from ray.tune import Analysis
import numpy as np
from tune_hyper import training_function
import json

analysis = Analysis("ray_results/training_function_2021-08-10_10-16-55")
print("Loaded")
df = analysis.dataframe(metric="mean_acc", mode="max")
largest_df = df.nlargest(20, 'mean_acc')

print(largest_df)

config_col = [col for col in df if col.startswith('config/')] + ['mean_acc']
configs_df = largest_df[config_col]

configs = configs_df.to_dict()
config_list = []
for id in configs['config/dp_1'].keys():
    config_list.append((configs['mean_acc'][id], {k.split('/')[1]: v[id] for k, v in configs.items() if 'config' in k}))

print(config_list)
sorted_all_results = sorted(config_list, key=lambda x: x[0], reverse=True)
print(sorted_all_results)
with open('result.json', 'w') as outfile:
    json.dump(sorted_all_results, outfile)


# all_results = []
# for i, config in enumerate(config_list):
#     result = []
#     for _ in range(5):
#         result.append(training_function(config))
#     all_results.append((np.mean(result), np.std(result), config))

