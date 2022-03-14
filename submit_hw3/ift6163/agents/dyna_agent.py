from collections import OrderedDict
from .base_agent import BaseAgent
from ift6163.models.ff_model import FFModel
from ift6163.policies.MLP_policy import MLPPolicyPG
from ift6163.critics.bootstrapped_continuous_critic import \
    BootstrappedContinuousCritic
from ift6163.infrastructure.replay_buffer import ReplayBuffer
from ift6163.infrastructure.utils import *


class MBAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super(MBAgent, self).__init__()

        self.env = env.unwrapped
        self.agent_params = agent_params
        self.gamma = self.agent_params['discount']
        self.reward_to_go = self.agent_params['reward_to_go']
        self.nn_baseline = self.agent_params['nn_baseline']
        self.ensemble_size = self.agent_params['ensemble_size']
        self.standardize_advantages = self.agent_params['standardize_advantages']

        self.dyn_models = []
        for i in range(self.ensemble_size):
            model = FFModel(
                self.agent_params['ac_dim'],
                self.agent_params['ob_dim'],
                self.agent_params['n_layers'],
                self.agent_params['size'],
                self.agent_params['learning_rate'],
            )
            self.dyn_models.append(model)

        # actor/policy
        self.actor = MLPPolicyPG(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            discrete=self.agent_params['discrete'],
            learning_rate=self.agent_params['learning_rate'],
            nn_baseline=self.agent_params['nn_baseline']
        )
        self.critic = BootstrappedContinuousCritic(self.agent_params)

        self.replay_buffer = ReplayBuffer()

    def train_actor(self, observations, actions, rewards_list, next_observations, terminals):

        """
            Training a PG agent refers to updating its actor using the given observations/actions
            and the calculated qvals/advantages that come from the seen rewards.
        """

        # TODO: update the PG actor/policy using the given batch of data, and
        # return the train_log obtained from updating the policy

        # HINT1: use helper functions to compute qvals and advantages
        # HINT2: look at the MLPPolicyPG class for how to update the policy
            # and obtain a train_log

        train_log = {}
        # print("training actor")
        # print("observations: ", len(observations))
        # print("actions: ", len(actions))
        # print("rewards_list: ", len(rewards_list))
        # print("next_observations: ", len(next_observations))

        # Compute the q-values from rewards
        q_values = self.calculate_q_vals(rewards_list)
        q_values = np.array(q_values)

        # Compute the advantages
        advantages = self.estimate_advantage(next_observations, rewards_list, q_values, terminals)

        # Transform advantages and q_values into a numpy array
        advantages = np.array(advantages)

        # Update the actor
        train_log = self.actor.update(observations, actions, advantages, q_values=q_values)

        return train_log

    def train_actor_critic(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        # TODO Implement the following pseudocode:
        # for agent_params['num_critic_updates_per_agent_update'] steps,
        #     update the critic

        # advantage = estimate_advantage(...)

        # for agent_params['num_actor_updates_per_agent_update'] steps,
        #     update the actor


        for i in range(self.agent_params['num_critic_updates_per_agent_update']):
            critic_loss = self.critic.update(ob_no, ac_na, next_ob_no, re_n, terminal_n)

        # advantage = self.estimate_advantage_critic(ob_no, next_ob_no, re_n, terminal_n)

        for i in range(self.agent_params['num_actor_updates_per_agent_update']):
            actor_loss = self.train_actor(ob_no, ac_na, re_n, next_ob_no, terminal_n)


        loss = OrderedDict()
        loss['Critic_Loss'] = critic_loss
        loss['Actor_Loss'] = actor_loss['Training_Loss']

        return actor_loss['Training_Loss'], critic_loss

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):

        # training a MB agent refers to updating the predictive model using observed state transitions
        # NOTE: each model in the ensemble is trained on a different random batch of size batch_size
        losses = []
        num_data = ob_no.shape[0]
        num_data_per_ens = int(num_data / self.ensemble_size)

        for i in range(self.ensemble_size):
            # select which datapoints to use for this model of the ensemble
            # you might find the num_data_per_env variable defined above useful

            # Copy this from previous homework
            start_index = i * num_data_per_ens
            end_index = (i + 1) * num_data_per_ens

            # observations = # TODO(Q1)
            observations = ob_no[start_index:end_index]
            # actions = # TODO(Q1)
            actions = ac_na[start_index:end_index]
            # next_observations = # TODO(Q1)
            next_observations = next_ob_no[start_index:end_index]

            # # use datapoints to update one of the dyn_models
            # model =  # TODO(Q1)
            model = self.dyn_models[i]
            log = model.update(observations, actions, next_observations,
                                self.data_statistics)
            loss = log['Training_Loss']
            losses.append(loss)

        # TODO Pick a model at random
        # TODO Use that model to generate one additional next_ob_no for every state in ob_no (using the policy distribution) 
        # Hint: You may need the env to label the rewards
        # Hint: Keep things on policy

        # Pick a random model
        random_model_index = np.random.randint(0, self.ensemble_size)
        random_model = self.dyn_models[random_model_index]

        # Use model to generate one additional next_ob_no for every state in ob_no (using the policy distribution)
        obs_next = random_model.get_prediction(ob_no, ac_na, self.data_statistics)

        # print("in TRAIN:")
        # print("obs_next: ", len(obs_next))
        # print("ob_no: ", len(ob_no))
        # print("ac_na: ", len(ac_na))
        # Get rewards for the ob_no and ac_na pairs
        rewards = self.env.get_reward(ob_no, ac_na)

        rewards = rewards[0]

        print("rewards: ", len(rewards))

        # TODO add this generated data to the real data
        path = Path(ob_no, [], ac_na, rewards, obs_next, terminal_n)
        paths = [path]
        self.add_to_replay_buffer(paths)


        # get a sample from new real data to train on
        batch_size = int(ob_no.shape[0] / self.ensemble_size)
        print("batch_size: ", batch_size)
        ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch = self.sample(batch_size)

        # TODO Perform a policy gradient update
        # Hint: Should the critic be trained with this generated data? Try with and without and include your findings in the report.
        actor_loss, critic_loss = self.train_actor_critic(ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch)

        loss = OrderedDict()
        loss['Critic_Loss'] = critic_loss
        loss['Actor_Loss'] = actor_loss
        loss['FD_Loss'] = np.mean(losses)

        # print("loss: ", loss['Actor_Loss'])
        # print("loss: ", loss['Critic_Loss'])
        # print("loss: ", loss['FD_Loss'])

        return loss

    def add_to_replay_buffer(self, paths, add_sl_noise=False):

        # add data to replay buffer
        self.replay_buffer.add_rollouts(paths, noised=add_sl_noise)

        # get updated mean/std of the data in our replay buffer
        self.data_statistics = {
            'obs_mean': np.mean(self.replay_buffer.obs, axis=0),
            'obs_std': np.std(self.replay_buffer.obs, axis=0),
            'acs_mean': np.mean(self.replay_buffer.acs, axis=0),
            'acs_std': np.std(self.replay_buffer.acs, axis=0),
            'delta_mean': np.mean(
                self.replay_buffer.next_obs - self.replay_buffer.obs, axis=0),
            'delta_std': np.std(
                self.replay_buffer.next_obs - self.replay_buffer.obs, axis=0),
        }

        # update the actor's data_statistics too, so actor.get_action can be calculated correctly
        self.actor.data_statistics = self.data_statistics

    def sample(self, batch_size):
        # NOTE: sampling batch_size * ensemble_size,
        # so each model in our ensemble can get trained on batch_size data
        return self.replay_buffer.sample_random_data(
            batch_size * self.ensemble_size)


    def calculate_q_vals(self, rewards_list):

        """
            Monte Carlo estimation of the Q function.
        """

        # TODO: return the estimated qvals based on the given rewards, using
            # either the full trajectory-based estimator or the reward-to-go
            # estimator

        # HINT1: rewards_list is a list of lists of rewards. Each inner list
            # is a list of rewards for a single trajectory.
        # HINT2: use the helper functions self._discounted_return and
            # self._discounted_cumsum (you will need to implement these). These
            # functions should only take in a single list for a single trajectory.

        # Case 1: trajectory-based PG
        # Estimate Q^{pi}(s_t, a_t) by the total discounted reward summed over entire trajectory
        # HINT3: q_values should be a 1D numpy array where the indices correspond to the same
        # ordering as observations, actions, etc.

        if not self.reward_to_go:
            # Compute the q_values for each trajectory
            q_values = []

            discounted = self._discounted_return(rewards_list)
            discounted_sum = np.sum(discounted)

            q_values += [discounted_sum]*len(rewards_list)

            # My custom test
            # q_values += discounted


        # Case 2: reward-to-go PG
        # Estimate Q^{pi}(s_t, a_t) by the discounted sum of rewards starting from t
        else:
            # Compute the q_values for each trajectory
            q_values = []

            discounted = self._discounted_cumsum(rewards_list)
            discounted_sum = np.sum(discounted)

            q_values += [discounted_sum]*len(rewards_list)

        return q_values

    def estimate_advantage(self, obs, rews_list, q_values, terminals):

        """
            Computes advantages by (possibly) using GAE, or subtracting a baseline from the estimated Q values
        """
        # print("In estimate_advantage")
        # print("obs: ", obs.shape)
        # print("q_values: ", q_values.ndim)

        # Estimate the advantage when nn_baseline is True,
        # by querying the neural network that you're using to learn the value function
        if self.nn_baseline:
            values_unnormalized = self.actor.run_baseline_prediction(obs)
            ## ensure that the value predictions and q_values have the same dimensionality
            ## to prevent silent broadcasting errors
            assert values_unnormalized.ndim == q_values.ndim
            ## TODO: values were trained with standardized q_values, so ensure
                ## that the predictions have the same mean and standard deviation as
                ## the current batch of q_values

            # Normalize the q_values
            q_values_mean = np.mean(q_values)
            q_values_std = np.std(q_values)
            q_values = (q_values - q_values_mean) / q_values_std

            # Normalize the value predictions
            values_mean = np.mean(values_unnormalized)
            values_std = np.std(values_unnormalized)
            values = (values_unnormalized - values_mean) / values_std


            # Note: added new config parameter to control whether to use GAE or not, callled gae
            if self.gae is True:
                ## append a dummy T+1 value for simpler recursive calculation
                values = np.append(values, [0])

                ## combine rews_list into a single array
                rews = np.concatenate(rews_list)

                ## create empty numpy array to populate with GAE advantage
                ## estimates, with dummy T+1 value for simpler recursive calculation
                batch_size = obs.shape[0]
                advantages = np.zeros(batch_size + 1)

                for i in reversed(range(batch_size)):
                    ## TODO: recursively compute advantage estimates starting from
                        ## timestep T.
                    ## HINT 1: use terminals to handle edge cases. terminals[i]
                        ## is 1 if the state is the last in its trajectory, and
                        ## 0 otherwise.
                    ## HINT 2: self.gae_lambda is the lambda value in the
                        ## GAE formula
                    # y=45 ## Remove: This is just to help with compiling

                    if terminals[i] == 1:
                        advantages[i] = rews[i] - values[i]

                    else:
                        advantages[i] = self.gae_lambda * self.gamma * advantages[i + 1] + rews[i] - values[i]


                # remove dummy advantage
                advantages = advantages[:-1]

            else:
                ## TODO: compute advantage estimates using q_values, and values as baselines
                advantages = q_values - values

        # Else, just set the advantage to [Q]
        else:
            advantages = q_values.copy()

        # Normalize the resulting advantages
        if self.standardize_advantages:
            ## TODO: standardize the advantages to have a mean of zero
            ## and a standard deviation of one

            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        return advantages

    def _discounted_return(self, rewards):
        """
            Helper function

            Input: list of rewards {r_0, r_1, ..., r_t', ... r_T} from a single rollout of length T

            Output: list where each index t contains sum_{t'=0}^T gamma^t' r_{t'}
        """

        # TODO: create list_of_discounted_returns
        list_of_discounted_returns = []

        for reward in rewards:
            discounted = 0

            for t_prime, _ in enumerate(rewards):
                discounted += self.gamma**t_prime * rewards[t_prime]

            list_of_discounted_returns.append(discounted)

        return list_of_discounted_returns

    def _discounted_cumsum(self, rewards):
        """
            Helper function which
            -takes a list of rewards {r_0, r_1, ..., r_t', ... r_T},
            -and returns a list where the entry in each index t' is sum_{t'=t}^T gamma^(t'-t) * r_{t'}
        """

        # TODO: create `list_of_discounted_returns`
        # HINT: it is possible to write a vectorized solution, but a solution
            # using a for loop is also fine

        list_of_discounted_cumsums = []

        for t, reward in enumerate(rewards):
            discounted = 0

            for t_prime, _ in enumerate(rewards):
                discounted += self.gamma**(t_prime-t) * rewards[t_prime]

            list_of_discounted_cumsums.append(discounted)

        return list_of_discounted_cumsums



    def estimate_advantage_critic(self, ob_no, next_ob_no, re_n, terminal_n):
        # TODO Implement the following pseudocode:
        # 1) query the critic with ob_no, to get V(s)
        # 2) query the critic with next_ob_no, to get V(s')
        # 3) estimate the Q value as Q(s, a) = r(s, a) + gamma*V(s')
        # HINT: Remember to cut off the V(s') term (ie set it to 0) at terminal states (ie terminal_n=1)
        # 4) calculate advantage (adv_n) as A(s, a) = Q(s, a) - V(s)

        values = self.critic.forward_np(ob_no)
        values_prime = self.critic.forward_np(next_ob_no)

        q_values = re_n + self.gamma * values_prime * (1 - terminal_n)

        adv_n = q_values - values

        if self.standardize_advantages:
            adv_n = (adv_n - np.mean(adv_n)) / (np.std(adv_n) + 1e-8)
        return adv_n