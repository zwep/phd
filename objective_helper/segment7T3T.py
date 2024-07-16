import numpy as np
import os
from loguru import logger
from objective_configuration.segment7T3T import DLOG
"""
Helper functions that are very specific to a task // project. Sadly we cant generalize everything

"""

logger.add(os.path.join(DLOG, 'objective_helper.log'))


def model_selection_processor(model_selection, model_name_list):
    logger.debug(f'\n\nProcessing the following model selection: {model_selection}')
    logger.debug('With model name list: ')
    logger.debug(model_selection)
    if model_selection.startswith('t'):
        logger.debug('Interpreting next strings as task numbers')
        # This is used to denote Task numbers
        sel_model_task = [str(x.strip()) for x in model_selection[1:].strip().split(',') if x]
        sel_model_name_list = []
        logger.debug(f'Found the tasks {sel_model_task}')
        for i_task in sel_model_task:
            logger.debug('Processing task ', i_task)
            if i_task.startswith('r'):
                # In some cases we need to write out the full 'raw' name.
                # Since we expect everything to start with a number, the chances are low that
                # this 'r' can be misinterpreted. But never zero.
                temp_name = [x for x in model_name_list if i_task[1:] == x]
                if len(temp_name):
                    temp_name = temp_name[0]
                else:
                    logger.error(f'We have not found this task {i_task}')
                    continue
            else:
                temp_name = [x for x in model_name_list if f'Task{i_task}' in x][0]
            sel_model_name_list.append(temp_name)
    else:
        # Assuming no task number, we go by the order that it was printed/found
        sel_model_index = np.array([int(x.strip()) for x in model_selection.strip().split(',') if x])
        sel_model_name_list = np.array(model_name_list)[sel_model_index]

    logger.debug(f'Returning the tasks {sel_model_name_list}')
    return list(sel_model_name_list)


