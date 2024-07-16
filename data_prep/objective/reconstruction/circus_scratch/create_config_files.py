import helper.misc as hmisc
import argparse
import os
import copy
from objective_configuration.reconstruction import DMODEL, DPRETRAINED, ANATOMY_LIST, \
    PERCENTAGE_LIST, DDATA, NUM_ITERATIONS_PARALLEL, CIRCUS_SCRATCH_APPENDIX

"""
We need a lot of config files...

All slightly different.

I think we can automate it.

--> This is for the parallel track...

"""


def update_inference(temp_yaml):
    # Deep copy is necessary
    single_val_dataset = copy.deepcopy(temp_yaml['validation']['datasets'][0])
    # Make sure we use CalagaryCampinas masking
    # This makes use of the np.ones() mask
    hmisc.set_nested(single_val_dataset, ['transforms', 'masking', 'name'], 'CalgaryCampinas')
    # And set a description
    hmisc.set_nested(single_val_dataset, ['text_description'], 'SebInference')
    # Append, so that the validation index can be set to 2
    temp_yaml['validation']['datasets'].append(single_val_dataset)
    return temp_yaml


def update_dict(yaml_dict, anatomical_region, percentage, batch_size=4, number_of_iterations=20000):
    """

    :param yaml_dict:
    :param anatomical_region: 2ch ,sa, transverse, 4ch
    :param percentage:
    :return:
    """
    # Avoid any typo's
    assert str(anatomical_region) in ANATOMY_LIST or anatomical_region == 'mixed'
    assert int(percentage) in PERCENTAGE_LIST

    # This can set the training list
    training_dir = os.path.join(DDATA, f'{anatomical_region}/train/train_{percentage}.lst')
    # Only use 20k iterations, convergence was then reached in most cases..
    hmisc.set_nested(yaml_dict, ['training', 'num_iterations'], number_of_iterations)
    # For all training and validation datasets, alter the following:
    for i_type in ['training', 'validation']:
        # Reduce this, because otherwise the GPU usages get too large...
        hmisc.set_nested(yaml_dict, [i_type, 'batch_size'], batch_size)
        for ii, i_dataset in enumerate(hmisc.get_nested(yaml_dict, [i_type, 'datasets'])):
            # This is needed to make sure that the further pipelining works..
            # Because index 0 and index 1 are linked to acc 5x and 10x.
            # This can be a possible source of error...
            if ii == 0:
                acceleration = 5
            else:
                acceleration = 10
            i_dataset['crop_outer_slices'] = False
            # This is not  CalagaryCampinas.
            # We have modified it such taht everything is radially sampled (interpolated)
            # Still use Radial, because otherwise we get a mask of ones
            hmisc.set_nested(i_dataset, ['name'], 'CalgaryCampinas')
            hmisc.set_nested(i_dataset, ['transforms', 'masking', 'name'], 'Radial')
            hmisc.set_nested(i_dataset, ['transforms', 'masking', 'accelerations'], [acceleration])
            # This is needed to detect the acceleration factor
            hmisc.set_nested(i_dataset, ['text_description'], f'{str(acceleration)}x')
            if i_type == 'training':
                i_dataset['filenames_lists'] = [training_dir]
                # Remove lists... We need filenames lists
                if 'lists' in i_dataset.keys():
                    del i_dataset['lists']
    return yaml_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-path', type=str, help='Provide the name of the directory that we want to post process. '
                                                'This is relative to the path ../pretrained_networks/direct ')

    """
    Circus mask   
    """

    p_args = parser.parse_args()
    path = p_args.path
    # To reduce memory usage - and make sure that we have enough epochs
    batch_size = 1

    original_config_yaml = os.path.join(DPRETRAINED, path, 'config.yaml')
    dest_path = os.path.join(DMODEL, path + CIRCUS_SCRATCH_APPENDIX, 'config')
    if not os.path.isdir(dest_path):
        os.makedirs(dest_path)

    # We now also have a 'mixed' anatomy class...
    i_anatomy = 'mixed'
    for i_percentage in PERCENTAGE_LIST:
        print('\t', i_percentage)
        # This now depends on how much data there is
        number_of_iterations = NUM_ITERATIONS_PARALLEL[i_percentage]
        dest_name = f'config_{i_percentage}p_{i_anatomy}.yaml'
        dest_name_inference = f'inference_config_{i_percentage}p_{i_anatomy}.yaml'
        ddest = os.path.join(dest_path, dest_name)
        ddest_inference = os.path.join(dest_path, dest_name_inference)
        # Always reload the original yaml
        temp_yaml = hmisc.load_yaml(original_config_yaml)
        update_dict(temp_yaml, anatomical_region=i_anatomy, percentage=i_percentage,
                    batch_size=batch_size, number_of_iterations=number_of_iterations)
        # Store the result..
        hmisc.write_yaml(temp_yaml, ddest)
        print('\t\t Written ', ddest)
        # We also write an inference config that is almost identical as the original one
        # This is a bit easier to modify, and we want it to be a separte file
        # So that it is not evaluated during training
        inference_yaml = copy.deepcopy(temp_yaml)
        update_inference(dict(inference_yaml))
        hmisc.write_yaml(inference_yaml, ddest_inference)
        print('\t\t Written ', ddest_inference)


#     for ii, i_dataset in enumerate(hmisc.get_nested(yaml_dict, [i_type, 'datasets'])):
#             # This is needed to make sure that the further pipelining works..
#             # Because index 0 and index 1 are linked to acc 5x and 10x.
#             # This can be a possible source of error...
#             if ii == 0:
#                 acceleration = 5
#             else:
#                 acceleration = 10
#             i_dataset['crop_outer_slices'] = False
#             hmisc.set_nested(i_dataset, ['transforms', 'masking', 'name'], 'Radial')