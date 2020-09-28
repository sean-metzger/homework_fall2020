print('Sko Buffs')
import subprocess
import shlex
import os 
import sys
from datetime import datetime

# datetime object containing current date and time
now = datetime.now()


bs  = [100]
lrs = [1,10]
for batch in bs: 
    for lr in lrs: 
        cs285_command = ('python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v2 \
        --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b %d -lr %.5f -rtg \
        --exp_name q2_b%d_r%.5f' %(batch, lr, batch, lr))

        print('command', cs285_command)
        basedir = '/userdata/smetzger/cs285/homework_fall2020/hw2/'
        times= now.strftime("%d_%m_%Y_%H_%M_%S")

        filename = basedir + '/runs/run' + ('_').join(cs285_command.split(' ')[2:]) + '.txt'
        string = "submit_job -q mind-gpu"
        string += " -m 78 -g 1"
        string += " -o " + filename
        string += ' -n deeprl'
        string += ' -x python '
        string += basedir + (' ').join(cs285_command.split(' ')[1:])

        print(string)
        cmd = shlex.split(string)
        print(cmd)
#         subprocess.run(shlex.split('export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-450'), stderr=subprocess.STDOUT)
#         subprocess.run(shlex.split('export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-440'), stderr=subprocess.STDOUT)
        subprocess.run(cmd, stderr=subprocess.STDOUT)