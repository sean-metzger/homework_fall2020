To replicate my results for questions 1, 2, and 4 run the commands for each question on the PDF here: 

http://rail.eecs.berkeley.edu/deeprlcourse/static/homeworks/hw3.pdf

For question 3, you will need to run the following commands: 

python run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q3_hparam1 --lr .001
python run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q3_hparam2 --lr .01
python run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q3_hparam3 --lr .0001
python run_hw3_dqn.py --env_name LunarLander-v3 --exp_name q3_hparam4 --lr .1

For question 5, use the following commands: 

python run_hw3_actor_critic.py --env_name InvertedPendulum-v2 --ep_len 1000 --discount
0.95 -n 100 -l 2 -s 64 -b 5000 -lr 0.01 --exp_name q5_1_100 -ntu 1 -ngsptu 100

python run_hw3_actor_critic.py --env_name HalfCheetah-v2 --ep_len 150 --discount 0.90 --
scalar_log_freq 1 -n 150 -l 2 -s 32 -b 30000 -eb 1500 -lr 0.02 --exp_name q5_1_100 -ntu 1 -ngsptu 100

Figures can be replicated by running make_graphs.ipynb
