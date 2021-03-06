print('Sko Buffs')
import subprocess
import shlex
import os 
import sys
from datetime import datetime

# datetime object containing current date and time
now = datetime.now()


bs  = [5]
lrs = [1e-2, 1e-1, 1]

stepss = [1]
for batch in bs: 
    for lr in lrs: 
        for steps in stepss: 
            cs285_command = ('python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v2 \
            --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b %d -lr %.5f -rtg \
            --exp_name q2_b%d_r%.5f' %(batch, lr, batch, lr))
    #         cs285_command = ('python cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 \
    # --discount 0.95 -n 100 -l 2 -s 32 -b %d -lr %.4f -rtg --nn_baseline \
    # --exp_name q4_search_b%d_lr%.4f_rtg_nnbaseline' %(batch, lr, batch, lr))

            print('command', cs285_command)
            basedir = '/userdata/smetzger/cs285/homework_fall2020/hw2/'
            times= now.strftime("%d_%m_%Y_%H_%M_%S")

            filename = basedir + '/runs/new' + ('_').join(cs285_command.split(' ')[2:]) + '.txt'
            string = "submit_job -q mind-gpu"
            string += " -m 78 -g 1"
            string += " -o " + filename
            string += ' -n deeprl'
            string += ' -x python '
            string += basedir + (' ').join(cs285_command.split(' ')[1:])

            print(string)
            cmd = shlex.split(string)
            print(cmd)
            subprocess.run(cmd, stderr=subprocess.STDOUT)
    
    
# print('Sko Buffs')
# import subprocess
# import shlex
# import os 
# import sys
# from datetime import datetime

# # datetime object containing current date and time
# now = datetime.now()


# bs  = [100, 200, 500, 1000]
# lrs = [5e-3, 7.5e-3, 1e-2]

# bs = [50000]
# lrs = [0.02]

# batch = bs[0]
# lr = lrs[0]
# commands = [
#     'python cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 \
# --discount 0.95 -n 100 -l 2 -s 32 -b <b*> -lr <r*> \
# --exp_name q4_b<b*>_r<r*>',
#     'python cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 \
# --discount 0.95 -n 100 -l 2 -s 32 -b <b*> -lr <r*> -rtg \
# --exp_name q4_b<b*>_r<r*>_rtg',
#     'python cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 \
# --discount 0.95 -n 100 -l 2 -s 32 -b <b*> -lr <r*> --nn_baseline \
# --exp_name q4_b<b*>_r<r*>_nnbaseline',
#     'python cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 \
# --discount 0.95 -n 100 -l 2 -s 32 -b <b*> -lr <r*> -rtg --nn_baseline \
# --exp_name q4_b<b*>_r<r*>_rtg_nnbaseline'
# ]
# for cs285_command in commands:

#         cs285_command = cs285_command.replace('<b*>', str(batch))
#         cs285_command = cs285_command.replace('<r*>', str(lr))
#         print('command', cs285_command)
#         basedir = '/userdata/smetzger/cs285/homework_fall2020/hw2/'
#         times= now.strftime("%d_%m_%Y_%H_%M_%S")

#         filename = basedir + '/runs/run' + ('_').join(cs285_command.split(' ')[2:]) + '.txt'
#         string = "submit_job -q mind-gpu"
#         string += " -m 78 -g 1"
#         string += " -o " + filename
#         string += ' -n deeprl'
#         string += ' -x python '
#         string += basedir + (' ').join(cs285_command.split(' ')[1:])

#         print(string)
#         cmd = shlex.split(string)
#         print(cmd)
# #         subprocess.run(shlex.split('export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-450'), stderr=subprocess.STDOUT)
# #         subprocess.run(shlex.split('export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-440'), stderr=subprocess.STDOUT)
#         subprocess.run(cmd, stderr=subprocess.STDOUT)