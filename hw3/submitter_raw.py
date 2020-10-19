print('Sko Buffs')
import subprocess
import shlex
import os 
import sys
from datetime import datetime

# datetime object containing current date and time
now = datetime.now()

cs285_command = (' ').join(sys.argv[1:])

print('command', cs285_command)
basedir = '/userdata/smetzger/cs285/homework_fall2020/hw3/'
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
subprocess.run(cmd, stderr=subprocess.STDOUT)