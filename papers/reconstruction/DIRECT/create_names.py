from objective_configuration.reconstruction import TYPE_NAMES, MODEL_NAMES


for i_type in TYPE_NAMES:
    print()
    for i_model in ['unet']:
        str_print = f"./papers/reconstruction/DIRECT/bundle_evaluate.sh -m {i_model}{i_type}"
        print(str_print)

sel_type = '_PRETR_SYNTH'
for i_model in MODEL_NAMES:
    inf_str1 = f'papers/reconstruction/DIRECT/inference_model.sh -m {i_model}{sel_type} -p 0 -o'
    inf_str2 = f'python data_postproc/objective/reconstruction/calculate_metrics_model.py -model {i_model}{sel_type} -p 0 --inference'
    eval_str1 = f'papers/reconstruction/DIRECT/evaluate_model.sh -m {i_model}{sel_type} -p 0 -o'
    eval_str2 = f'python data_postproc/objective/reconstruction/calculate_metrics_model.py -model {i_model}{sel_type} -p 0'
    print('\n')
    print(inf_str1)
    print(inf_str2)
    print(eval_str1)
    print(eval_str2)

sel_type = '_PRETR_SYNTH_ACQ'
for i_model in MODEL_NAMES:
    print(i_model)
    str1 = f"papers/reconstruction/DIRECT/bundle_train_scripts/bundle_train_pretr_synth_acq.sh -m {i_model}"
    str2 = f"./papers/reconstruction/DIRECT/bundle_inference.sh -m {i_model}{sel_type}"
    str3 = f"./papers/reconstruction/DIRECT/bundle_evaluate.sh -m {i_model}{sel_type}"
    print(str1)
    print(str2)
    print(str3)

# Easily setup calc metric and vis metric...
for i_type in TYPE_NAMES:
    print()
    for i_model in ['unet']:
        model_type = f'{i_model}{i_type}'
        calc_str = f'python data_postproc/objective/reconstruction/calculate_metrics_model.py -m {model_type}'
        vis_str = f'python data_postproc/objective/reconstruction/visualize_metrics_model.py -m {model_type}'
        print(calc_str)
        # print(vis_str)

# Redo the evaluation on 0p. So first remove everything
from objective_configuration.reconstruction import DRESULT
import os
import shutil
for i_type in TYPE_NAMES:
    print()
    for i_model in MODEL_NAMES:
        model_type = f'{i_model}{i_type}'
        dest_path = os.path.join(DRESULT, model_type, '0p')
        print(dest_path)
        shutil.rmtree(dest_path)


# Redo the evaluation on 0p. So first remove everything
from objective_configuration.reconstruction import DRESULT, TYPE_NAMES, MODEL_NAMES, DRESULT_INFERENCE
import os
for i_type in TYPE_NAMES:
    print()
    for i_model in MODEL_NAMES:
        model_type = f'{i_model}{i_type}'
        dest_path = os.path.join(DRESULT, model_type)
        print(dest_path)
        os.makedirs(dest_path, exist_ok=True)
        dest_path = os.path.join(DRESULT_INFERENCE, model_type)
        print(dest_path)
        os.makedirs(dest_path, exist_ok=True)

# Now try to get all the metrics jsons
from objective_configuration.reconstruction import DRESULT, TYPE_NAMES, MODEL_NAMES, DRESULT_INFERENCE
import os

for i_type in TYPE_NAMES:
    print()
    for i_model in MODEL_NAMES:
        model_type = f'{i_model}{i_type}'
        dest_path = os.path.join(DRESULT_INFERENCE, model_type, 'metric.json')
        source_path = f'sharreve@rtrspla01:/local_scratch/sharreve/paper/reconstruction/inference/{model_type}/metric.json'
        scp_command = f' sshpass -p "QzYzTgzxcYVxD5C2" scp {source_path} {dest_path}'
        try:
            os.system(scp_command)
        except:
            pass
        # dest_path = os.path.join(DRESULT, model_type, 'metric.json')
        # source_path = f'sharreve@rtrspla01:/local_scratch/sharreve/paper/reconstruction/results/{model_type}/metric.json'
        # scp_command = f' sshpass -p "QzYzTgzxcYVxD5C2" scp {source_path} {dest_path}'
        # try:
        #     os.system(scp_command)
        # except:
        #     pass
        # os.makedirs(dest_path)

