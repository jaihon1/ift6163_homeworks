import os
import time

import sys
print(sys.path)


from ift6163.agents.dqn_agent import DQNAgent
from ift6163.agents.ddpg_agent import DDPGAgent
from ift6163.agents.td3_agent import TD3Agent
from ift6163.infrastructure.rl_trainer import RL_Trainer
import hydra, json
from omegaconf import DictConfig, OmegaConf
from ift6163.infrastructure.dqn_utils import get_env_kwargs

class offpolicy_Trainer(object):

    def __init__(self, params):

        #####################
        ## SET AGENT PARAMS
        #####################

        self.params = params
        self.params['batch_size_initial'] = self.params['batch_size']

        if self.params['rl_alg'] == 'dqn':
            agent = DQNAgent
        elif self.params['rl_alg'] == 'ddpg':
            agent = DDPGAgent
        elif self.params['rl_alg'] == 'td3':
            agent = TD3Agent
        else:
            print("Pick a rl_alg first")
            sys.exit()
        print(self.params)
        print(self.params['train_batch_size'])

        ################
        ## RL TRAINER
        ################

        self.rl_trainer = RL_Trainer(self.params , agent_class =  agent)

    def run_training_loop(self):

        self.rl_trainer.run_training_loop(
            self.params['n_iter'],
            collect_policy = self.rl_trainer.agent.actor,
            eval_policy = self.rl_trainer.agent.actor,
            )


@hydra.main(config_path="conf", config_name="config")
def my_main(cfg: DictConfig):
    my_app(cfg)


def my_app(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    import os
    print("Command Dir:", os.getcwd())
    # print ("params: ", json.dumps(params, indent=4))
    if cfg['env_name']=='reacher-ift6163-v0':
        cfg['ep_len']=200
    if cfg['env_name']=='cheetah-ift6163-v0':
        cfg['ep_len']=500
    if cfg['env_name']=='obstacles-ift6163-v0':
        cfg['ep_len']=100

    params = vars(cfg)
    # params.extend(env_args)
    for key, value in cfg.items():
        params[key] = value
    if cfg['atari']:
        env_args = get_env_kwargs(cfg['env_name'])
        for key, value in env_args.items():
            params[key] = value
    print ("params: ", params)

    ##################################
    ### CREATE DIRECTORY FOR LOGGING
    ##################################


    logdir_prefix = 'hw3_'  # keep for autograder

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    logdir = logdir_prefix + cfg.exp_name + '_' + cfg.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    params['logdir'] = logdir
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)
    from omegaconf import open_dict
    with open_dict(cfg):
        cfg.logdir = logdir

    print("\n\n\nLOGGING TO: ", logdir, "\n\n\n")

    ###################
    ### RUN TRAINING
    ###################
    # cfg = OmegaConf.merge(cfg, params)
    trainer = offpolicy_Trainer(params)
    trainer.run_training_loop()



if __name__ == "__main__":
    import os
    print("Command Dir:", os.getcwd())
    my_main()



# Question 1:

# python run_hw4.py env_name=MsPacman-v0 exp_name=q1 scalar_log_freq=1000 n_iter=300000 double_q=False
# python run_hw4.py env_name=LunarLander-v3 exp_name=q1 scalar_log_freq=1000 n_iter=300000 double_q=False # used for testing

# Question 2:
# python run_hw4.py env_name=LunarLander-v3 exp_name=q2_dqn_1 seed=1 scalar_log_freq=1000 n_iter=300000 double_q=False
# python run_hw4.py env_name=LunarLander-v3 exp_name=q2_dqn_2 seed=2 scalar_log_freq=1000 n_iter=300000 double_q=False
# python run_hw4.py env_name=LunarLander-v3 exp_name=q2_dqn_3 seed=3 scalar_log_freq=1000 n_iter=300000 double_q=False

# python run_hw4.py env_name=LunarLander-v3 exp_name=q2_doubledqn_1 double_q=true seed=1 scalar_log_freq=1000 n_iter=300000
# python run_hw4.py env_name=LunarLander-v3 exp_name=q2_doubledqn_2 double_q=true seed=2 scalar_log_freq=1000 n_iter=300000
# python run_hw4.py env_name=LunarLander-v3 exp_name=q2_doubledqn_3 double_q=true seed=3 scalar_log_freq=1000 n_iter=300000


# Question 3:

# python run_hw4.py env_name=LunarLander-v3 exp_name=q3_gamma50 gamma=0.50 scalar_log_freq=1000 n_iter=300000 double_q=False
# python run_hw4.py env_name=LunarLander-v3 exp_name=q3_gamma90 gamma=0.90 scalar_log_freq=1000 n_iter=300000 double_q=False
# python run_hw4.py env_name=LunarLander-v3 exp_name=q3_gamma95 gamma=0.95 scalar_log_freq=1000 n_iter=300000 double_q=False
# python run_hw4.py env_name=LunarLander-v3 exp_name=q3_gamma99 gamma=0.99 scalar_log_freq=1000 n_iter=300000 double_q=False

# python run_hw4.py env_name=LunarLander-v3 exp_name=q3_lr1e-1 learning_rate=1e-1 critic_learning_rate=1e-1 scalar_log_freq=1000 n_iter=300000 double_q=False
# python run_hw4.py env_name=LunarLander-v3 exp_name=q3_lr1e-2 learning_rate=1e-2 critic_learning_rate=1e-2 scalar_log_freq=1000 n_iter=300000 double_q=False
# python run_hw4.py env_name=LunarLander-v3 exp_name=q3_lr1e-3 learning_rate=1e-3 critic_learning_rate=1e-3 scalar_log_freq=1000 n_iter=300000 double_q=False
# python run_hw4.py env_name=LunarLander-v3 exp_name=q3_lr1e-4 learning_rate=1e-4 critic_learning_rate=1e-4 scalar_log_freq=1000 n_iter=300000 double_q=False


# Question 4:
# python run_hw4.py exp_name=q4_ddpg_up_lr1e-1 rl_alg=ddpg env_name=InvertedPendulum-v2 atari=false learning_rate=1e-1 critic_learning_rate=1e-1 learning_freq=1 scalar_log_freq=1000 n_iter=50000
# python run_hw4.py exp_name=q4_ddpg_up_lr1e-2 rl_alg=ddpg env_name=InvertedPendulum-v2 atari=false learning_rate=1e-2 critic_learning_rate=1e-2 learning_freq=1 scalar_log_freq=1000 n_iter=50000
# python run_hw4.py exp_name=q4_ddpg_up_lr1e-3 rl_alg=ddpg env_name=InvertedPendulum-v2 atari=false learning_rate=1e-3 critic_learning_rate=1e-3 learning_freq=1 scalar_log_freq=1000 n_iter=50000
# python run_hw4.py exp_name=q4_ddpg_up_lr1e-4 rl_alg=ddpg env_name=InvertedPendulum-v2 atari=false learning_rate=1e-3 critic_learning_rate=1e-4 learning_freq=1 scalar_log_freq=1000 n_iter=50000

# python run_hw4.py exp_name=q4_ddpg_up_tuf1 rl_alg=ddpg env_name=InvertedPendulum-v2 atari=false learning_rate=1e-2 critic_learning_rate=1e-2 learning_freq=1 scalar_log_freq=1000 n_iter=50000
# python run_hw4.py exp_name=q4_ddpg_up_tuf2 rl_alg=ddpg env_name=InvertedPendulum-v2 atari=false learning_rate=1e-2 critic_learning_rate=1e-2 learning_freq=2 scalar_log_freq=1000 n_iter=50000
# python run_hw4.py exp_name=q4_ddpg_up_tuf4 rl_alg=ddpg env_name=InvertedPendulum-v2 atari=false learning_rate=1e-2 critic_learning_rate=1e-2 learning_freq=4 scalar_log_freq=1000 n_iter=50000
# python run_hw4.py exp_name=q4_ddpg_up_tuf8 rl_alg=ddpg env_name=InvertedPendulum-v2 atari=false learning_rate=1e-2 critic_learning_rate=1e-2 learning_freq=8 scalar_log_freq=1000 n_iter=50000

# Best results:

# learning_rate=1e-2 critic_learning_rate=1e-2
# learning_freq=4

# Question 5:
# python run_hw4.py exp_name=q5_ddpg_hard_uplf2_lr1e-2 rl_alg=ddpg env_name=HalfCheetah-v2 atari=false learning_rate=1e-2 critic_learning_rate=1e-2 learning_freq=4 scalar_log_freq=1000 n_iter=150000 size_hidden_critic=128 size=128


# Question 6:
# python run_hw4.py exp_name=q6_td3_rho0.1 rl_alg=td3 env_name=InvertedPendulum-v2 atari=false td3_target_policy_noise=0.1 learning_rate=1e-2 critic_learning_rate=1e-2 learning_freq=1 scalar_log_freq=1000 n_iter=50000
# python run_hw4.py exp_name=q6_td3_rho0.2 rl_alg=td3 env_name=InvertedPendulum-v2 atari=false td3_target_policy_noise=0.2 learning_rate=1e-2 critic_learning_rate=1e-2 learning_freq=1 scalar_log_freq=1000 n_iter=50000
# python run_hw4.py exp_name=q6_td3_rho0.4 rl_alg=td3 env_name=InvertedPendulum-v2 atari=false td3_target_policy_noise=0.4 learning_rate=1e-2 critic_learning_rate=1e-2 learning_freq=1 scalar_log_freq=1000 n_iter=50000
# python run_hw4.py exp_name=q6_td3_rho0.8 rl_alg=td3 env_name=InvertedPendulum-v2 atari=false td3_target_policy_noise=0.8 learning_rate=1e-2 critic_learning_rate=1e-2 learning_freq=1 scalar_log_freq=1000 n_iter=50000


# python run_hw4.py exp_name=q6_td3_tuf1 rl_alg=td3 env_name=InvertedPendulum-v2 atari=false target_update_freq=1 actor_update_freq=1 td3_target_policy_noise=0.2 learning_rate=1e-2 critic_learning_rate=1e-2 learning_freq=1 scalar_log_freq=1000 n_iter=50000
# python run_hw4.py exp_name=q6_td3_tuf2 rl_alg=td3 env_name=InvertedPendulum-v2 atari=false target_update_freq=2 actor_update_freq=2 td3_target_policy_noise=0.2 learning_rate=1e-2 critic_learning_rate=1e-2 learning_freq=1 scalar_log_freq=1000 n_iter=50000
# python run_hw4.py exp_name=q6_td3_tuf5 rl_alg=td3 env_name=InvertedPendulum-v2 atari=false target_update_freq=5 actor_update_freq=5 td3_target_policy_noise=0.2 learning_rate=1e-2 critic_learning_rate=1e-2 learning_freq=1 scalar_log_freq=1000 n_iter=50000
# python run_hw4.py exp_name=q6_td3_tuf10 rl_alg=td3 env_name=InvertedPendulum-v2 atari=false target_update_freq=10 actor_update_freq=10 td3_target_policy_noise=0.2 learning_rate=1e-2 critic_learning_rate=1e-2 learning_freq=1 scalar_log_freq=1000 n_iter=50000
# python run_hw4.py exp_name=q6_td3_tuf50 rl_alg=td3 env_name=InvertedPendulum-v2 atari=false target_update_freq=50 actor_update_freq=50 td3_target_policy_noise=0.2 learning_rate=1e-2 critic_learning_rate=1e-2 learning_freq=1 scalar_log_freq=1000 n_iter=50000
# python run_hw4.py exp_name=q6_td3_tuf100 rl_alg=td3 env_name=InvertedPendulum-v2 atari=false target_update_freq=100 actor_update_freq=100 td3_target_policy_noise=0.2 learning_rate=1e-2 critic_learning_rate=1e-2 learning_freq=1 scalar_log_freq=1000 n_iter=50000

# Best results:
# td3_target_policy_noise=0.2
# target_update_freq=5
# actor_update_freq=5


# Question 7:

# python run_hw4.py exp_name=q7_td3_tuf10_rho0.2 rl_alg=td3 env_name=HalfCheetah-v2 atari=false target_update_freq=5 actor_update_freq=5 td3_target_policy_noise=0.2 learning_rate=1e-2 critic_learning_rate=1e-2 learning_freq=4 scalar_log_freq=1000 n_iter=150000 size_hidden_critic=128 size=128