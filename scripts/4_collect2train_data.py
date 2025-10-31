import os


root_dir = "/home/zexi/Dev/dobot_xtrainer_decompiled/dobot_xtrainer/datasets"
dataset_name = "dobot_pick_random_1012"

dataset_dir = root_dir + "/" + dataset_name + "/collect_data/"
all_data_dir = os.listdir(dataset_dir)
all_data_dir.sort(key=lambda x: int(x))
print(all_data_dir)

idx = 0
for i in all_data_dir:
    print("dealing with: ", i)
    CMD = [
        'python', "script_collect2train.py",
        '--root_dir', str(root_dir),
        '--dataset_name', str(dataset_name),
        '--date_collect', i,
        '--idx', str(idx)
    ]
    rt_code = os.system(" ".join(CMD))
    if rt_code:
        break
    idx+=1