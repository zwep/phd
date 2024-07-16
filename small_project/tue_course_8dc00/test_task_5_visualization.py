import helper.plot_class as hplotc
import helper.misc as hmisc
import matplotlib.pyplot as plt
import json
import sys
import os
import numpy as np

username = os.environ.get('USER', os.environ.get('USERNAME'))
remote = False
if username != 'bugger':
    remote = True

data_path_a = '/home/bugger/Documents/TUE/AssistentDocent/Assignment2/task_5_a.json'
data_path_b = '/home/bugger/Documents/TUE/AssistentDocent/Assignment2/task_5_b.json'
ddest = '/home/bugger/Documents/TUE/AssistentDocent/Assignment2'

if remote:
    data_path_a = '/data/seb/course_8dc00/task_5_a.json'
    data_path_b = '/data/seb/course_8dc00/task_5_b.json'
    ddest = '/data/seb/course_8dc00'

with open(data_path_a, 'r') as f:
    A = f.read()

a_split = A.split('learning_rate')
import re
result_dict_list = []
for ii, i_str in enumerate(a_split[1:-1]):
    new_str = '{"learning_rate' + i_str
    new_new_str = re.sub('\}\{"', "}", new_str)
    new_new_str = re.sub('\}, \{"', "}", new_new_str)
    parsed_dict = json.loads(new_new_str)
    result_dict_list.append(parsed_dict)

result_dict_list = sorted(result_dict_list, key=lambda x: x['test_loss'])

result_test_losses = [x['test_loss'] for x in result_dict_list]
result_lr = [x['learning_rate'] for x in result_dict_list]
result_batch_size = [x['batch_size'] for x in result_dict_list]
result_epoch = [x['epoch'] for x in result_dict_list]
result_val_loss = [np.mean(x['val_loss']) for x in result_dict_list]

fig, ax = plt.subplots(2, 3, figsize=(20, 20))
ax = ax.ravel()
ax[0].plot(result_test_losses)
ax[0].set_title('test loss, sorted by test loss')
ax[1].plot(result_lr)
ax[1].set_title('learning rate, sorted by test loss')
ax[2].plot(result_batch_size)
ax[2].set_title('batch size, sorted by test loss')
ax[3].plot(result_epoch)
ax[3].set_title('epoch, sorted by test loss')
ax[4].plot(result_val_loss)
ax[4].set_title('val loss, sorted by test loss')
fig.savefig(os.path.join(ddest, 'overview_metrics_all.png'))


