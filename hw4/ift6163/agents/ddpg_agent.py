import numpy as np
import torch

from ift6163.infrastructure.replay_buffer import ReplayBuffer
from ift6163.infrastructure.dqn_utils import MemoryOptimizedReplayBuffer
from ift6163.policies.MLP_policy import MLPPolicyDeterministic
from ift6163.critics.ddpg_critic import DDPGCritic
from ift6163.infrastructure import pytorch_util as ptu
import copy

class DDPGAgent(object):
    def __init__(self, env, agent_params):

        self.env = env
        self.agent_params = agent_params
        print ("agent_params", agent_params)
        self.batch_size = agent_params['train_batch_size']
        # import ipdb; ipdb.set_trace()
        self.last_obs = self.env.reset()

        self.num_actions = self.env.action_space.shape[0]
        self.learning_starts = agent_params['learning_starts']
        self.learning_freq = agent_params['learning_freq']
        self.target_update_freq = agent_params['target_update_freq']
        self.exploration = agent_params['exploration_schedule']

        self.replay_buffer_idx = None
        self.optimizer_spec = agent_params['optimizer_spec']

        self.actor = MLPPolicyDeterministic(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            discrete=self.agent_params['discrete'],
            learning_rate=self.agent_params['learning_rate'],
            nn_baseline=False
        )
        ## Create the Q function
        self.q_fun = DDPGCritic(self.actor, agent_params, self.optimizer_spec)

        ## Hint: We can use the Memory optimized replay buffer but now we have continuous actions
        self.replay_buffer = MemoryOptimizedReplayBuffer(
            agent_params['replay_buffer_size'], agent_params['frame_history_len'], lander=True,
            continuous_actions=True, ac_dim=self.agent_params['ac_dim'])
        self.t = 0
        self.num_param_updates = 0

    def add_to_replay_buffer(self, paths):
        pass

    def step_env(self):
        """
            Step the env and store the transition
            At the end of this block of code, the simulator should have been
            advanced one step, and the replay buffer should contain one more transition.
            Note that self.last_obs must always point to the new latest observation.
        """

        # TODO store the latest observation ("frame") into the replay buffer
        # HINT: the replay buffer used here is `MemoryOptimizedReplayBuffer`
            # in dqn_utils.py
        latest_observation_frame = self.last_obs
        self.replay_buffer_idx = self.replay_buffer.store_frame(latest_observation_frame)

        # # TODO add noise to the deterministic policy
        # perform_random_action = TODO
        # # HINT: take random action
        # action = TODO

        # TODO use epsilon greedy exploration when selecting action
        perform_random_action = False

        if self.t < self.learning_starts:
            perform_random_action = True

        if perform_random_action:
            # HINT: take random action
                # with probability eps (see np.random.random())
                # OR if your current step number (see self.t) is less that self.learning_starts
            action = self.env.action_space.sample()
        else:
            # HINT: Your actor will take in multiple previous observations ("frames") in order
                # to deal with the partial observability of the environment. Get the most recent
                # `frame_history_len` observations using functionality from the replay buffer,
                # and then use those observations as input to your actor.

            observation = self.replay_buffer.encode_recent_observation()
            action = self.actor(ptu.from_numpy(observation))

            # add noise to the deterministic policy
            noise_weight = 0.10
            noise = torch.randn(action.shape) * noise_weight
            action = action + noise

            # keep action in range of lowest action and highest action
            low_action = self.env.action_space.low[0]
            high_action = self.env.action_space.high[0]
            # print("action", action)
            action = torch.clamp(action, low_action, high_action)

            action = ptu.to_numpy(action)

        # TODO take a step in the environment using the action from the policy
        # HINT1: remember that self.last_obs must always point to the newest/latest observation
        # HINT2: remember the following useful function that you've seen before:
            #obs, reward, done, info = env.step(action)
        obs, reward, done, info = self.env.step(action)
        self.last_obs = obs

        # TODO store the result of taking this action into the replay buffer
        # HINT1: see your replay buffer's `store_effect` function
        # HINT2: one of the arguments you'll need to pass in is self.replay_buffer_idx from above
        self.replay_buffer.store_effect(self.replay_buffer_idx, action, reward, done)

        # TODO if taking this step resulted in done, reset the env (and the latest observation)
        if done:
            self.last_obs = self.env.reset()


    def sample(self, batch_size):
        if self.replay_buffer.can_sample(self.batch_size):
            return self.replay_buffer.sample(batch_size)
        else:
            return [],[],[],[],[]

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        log = {}
        if (self.t > self.learning_starts
                and self.t % self.learning_freq == 0
                and self.replay_buffer.can_sample(self.batch_size)
        ):

            # TODO fill in the call to the update function using the appropriate tensors
            log = self.q_fun.update(
                ob_no, ac_na, next_ob_no, re_n, terminal_n
            )

            # TODO fill in the call to the update function using the appropriate tensors
            ## Hint the actor will need a copy of the q_net to maximize the Q-function
            loss_actor = self.actor.update(ob_no, self.q_fun.q_net)
            log.update({'Training_Loss_Actor': loss_actor})

            # TODO update the target network periodically
            # HINT: your critic already has this functionality implemented
            if self.num_param_updates % self.target_update_freq == 0:
                self.q_fun.update_target_network()

            self.num_param_updates += 1

        self.t += 1
        return log
