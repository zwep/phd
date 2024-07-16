import argparse
import os
import objective.pinn_efields.executor_pinn_efields as executor
import objective.pinn_efields.postproc_pinn_efields as postproc


parser = argparse.ArgumentParser()
parser.add_argument('-model_name', type=str)
parser.add_argument('-config', type=str, default='00')


# Parses the input
p_args = parser.parse_args()
model_name = p_args.model_name
config = p_args.config

ddata = '/local_scratch/sharreve/mri_data/pinn_fdtd'
biasfield_model_path = f'/local_scratch/sharreve/model_run/{model_name}/config_{config}'
ddest = f'/local_scratch/sharreve/model_run/{model_name}/model_results'
if not os.path.isdir(ddest):
       os.makedirs(ddest)

postproc_obj = postproc.PostProcPinnEfields(executor_module=executor, ddest=ddest,
                                            config_name='config_param.json',
                                            config_path=biasfield_model_path,
                                            debug=True)

postproc_obj.visualize_test_example(5)
postproc_obj.evaluate_test_loader(n_examples=10)
postproc_obj.store_single_result(index=0)

