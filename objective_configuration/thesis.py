import os

username = os.environ.get('USER', os.environ.get('USERNAME'))
remote = False
if username != 'bugger':
    remote = True

if remote:
    DPLOT = '/local_scratch/sharreve'
else:
    DPLOT = '/home/bugger/Documents/paper/thesis/Thesis/Figures'