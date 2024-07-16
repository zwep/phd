import helper.misc as hmisc
import argparse
import os
import copy
from objective_configuration.reconstruction import DMODEL, DPRETRAINED, ANATOMY_LIST, \
    PERCENTAGE_LIST, DDATA_SYNTH, NUM_ITERATIONS_PARALLEL, PRETR_SYNTH_APPENDIX

from objective_helper.reconstruction import DIRECT_update_config_dict, DIRECT_update_inference_dict


"""
We need a lot of config files...

All slightly different.

I think we can automate it.

--> This is for the parallel track...

"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-path', type=str, help='Provide the name of the directory that we want to post process. '
                                                'This is relative to the path ../pretrained_networks/direct ')


    p_args = parser.parse_args()
    path = p_args.path
    # To reduce memory usage - and make sure that we have enough epochs
    if 'unet' in path:
        # For unet we increase the batch size
        batch_size = 1
    else:
        batch_size = 1

    original_config_yaml = os.path.join(DPRETRAINED, path, 'config.yaml')
    dest_path = os.path.join(DMODEL, path + PRETR_SYNTH_APPENDIX, 'config')
    if not os.path.isdir(dest_path):
        os.makedirs(dest_path)

    # We now also have a 'mixed' anatomy class...
    i_anatomy = 'mixed'
    for i_percentage in PERCENTAGE_LIST:
        print('\t', i_percentage)
        # This now depends on how much data there is
        number_of_iterations = int(NUM_ITERATIONS_PARALLEL[i_percentage] / batch_size)
        dest_name = f'config_{i_percentage}p_{i_anatomy}.yaml'
        dest_name_inference = f'inference_config_{i_percentage}p_{i_anatomy}.yaml'
        ddest = os.path.join(dest_path, dest_name)
        ddest_inference = os.path.join(dest_path, dest_name_inference)
        # Always reload the original yaml
        temp_yaml = hmisc.load_yaml(original_config_yaml)
        DIRECT_update_config_dict(temp_yaml, ddata=DDATA_SYNTH, anatomical_region=i_anatomy, percentage=i_percentage,
                                  batch_size=batch_size, number_of_iterations=number_of_iterations)
        # Store the result..
        hmisc.write_yaml(temp_yaml, ddest)
        print('\t\t Written ', ddest)
        # We also write an inference config that is almost identical as the original one
        # This is a bit easier to modify, and we want it to be a separate file
        # So that it is not evaluated during training
        inference_yaml = copy.deepcopy(temp_yaml)
        DIRECT_update_inference_dict(dict(inference_yaml))
        hmisc.write_yaml(inference_yaml, ddest_inference)
        print('\t\t Written ', ddest_inference)

