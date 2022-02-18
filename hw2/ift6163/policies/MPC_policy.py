from random import random
import numpy as np

from .base_policy import BasePolicy


class MPCPolicy(BasePolicy):

    def __init__(self,
                 env,
                 ac_dim,
                 dyn_models,
                 horizon,
                 N,
                 sample_strategy='random',
                 cem_iterations=4,
                 cem_num_elites=5,
                 cem_alpha=1,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.env = env
        self.dyn_models = dyn_models
        self.horizon = horizon
        self.N = N
        self.data_statistics = None  # NOTE must be updated from elsewhere

        self.ob_dim = self.env.observation_space.shape[0]

        # action space
        self.ac_space = self.env.action_space
        self.ac_dim = ac_dim
        self.low = self.ac_space.low
        self.high = self.ac_space.high

        # Sampling strategy
        allowed_sampling = ('random', 'cem')
        assert sample_strategy in allowed_sampling, f"sample_strategy must be one of the following: {allowed_sampling}"
        self.sample_strategy = sample_strategy
        self.cem_iterations = cem_iterations
        self.cem_num_elites = cem_num_elites
        self.cem_alpha = cem_alpha

        print(f"Using action sampling strategy: {self.sample_strategy}")
        if self.sample_strategy == 'cem':
            print(f"CEM params: alpha={self.cem_alpha}, "
                + f"num_elites={self.cem_num_elites}, iterations={self.cem_iterations}")

    def sample_action_sequences(self, num_sequences, horizon, obs=None):
        # print(f"Sampling {num_sequences} action sequences of length {horizon}")
        # print(f"Action space: {self.ac_space}")
        # print(f"Action dimension: {self.ac_dim}")
        # print(f"Action space low: {self.low}")
        # print(f"Action space high: {self.high}")

        if self.sample_strategy == 'random' \
            or (self.sample_strategy == 'cem' and obs is None):
            # TODO(Q1) uniformly sample trajectories and return an array of
            # dimensions (num_sequences, horizon, self.ac_dim) in the range
            # [self.low, self.high]
            # Hint: you can use np.random.uniform to sample uniformly from a range of values (Q1) or use the following code (Q2) to sample from a uniform distribution in the action space

            # For every sequence (random_action_sequence[i]), sample an action sequence of length horizon
            random_action_sequences = np.random.uniform(low=self.low, high=self.high, size=(num_sequences, horizon, self.ac_dim))

            # print(f"Random action sequences: {random_action_sequences.shape}")


            return random_action_sequences

        elif self.sample_strategy == 'cem':
            # TODO(Q5): Implement action selection using CEM.
            # Begin with randomly selected actions, then refine the sampling distribution
            # iteratively as described in Section 3.3, "Iterative Random-Shooting with Refinement" of
            # https://arxiv.org/pdf/1909.11652.pdf
            elite_mean = 0.0
            elite_variance = 1.0
            elite_action_sequences = np.zeros((self.cem_num_elites, self.horizon, self.ac_dim))

            for i in range(self.cem_iterations):
                # - Sample candidate sequences from a Gaussian with the current
                #   elite mean and variance
                #     (Hint: remember that for the first iteration, we instead sample
                #      uniformly at random just like we do for random-shooting)
                # - Get the top `self.cem_num_elites` elites
                #     (Hint: what existing function can we use to compute rewards for
                #      our candidate sequences in order to rank them?)
                # - Update the elite mean and variance

                action_sequences = np.random.normal(loc=elite_mean, scale=elite_variance, size=(num_sequences, horizon, self.ac_dim))

                # print(f"Action sequences: {action_sequences.shape}")

                # Calculate rewards for each candidate sequence
                rewards = self.evaluate_candidate_sequences(action_sequences, obs)

                # print(f"Rewards: {rewards}")

                # Sort the candidate sequences by rewards
                sorted_indices = np.argsort(rewards)[::-1]

                # print(f"Sorted indices: {sorted_indices}")

                # Get the top `self.cem_num_elites` elite sequences
                elite_action_sequences = action_sequences[sorted_indices[:self.cem_num_elites]]

                # print(f"Elite action sequences: {elite_action_sequences.shape}")

                # Update the elite mean and variance
                elite_mean = np.mean(elite_action_sequences, axis=0)
                elite_variance = np.var(elite_action_sequences, axis=0)

                # print(f"cem iteration {i}")
                # print(f"Elite mean: {elite_mean}")
                # print(f"Elite variance: {elite_variance}")

            # TODO(Q5): Set `cem_action` to the appropriate action sequence chosen by CEM.
            # The shape should be (horizon, self.ac_dim)
            cem_action = elite_action_sequences[0]

            return cem_action[None]
        else:
            raise Exception(f"Invalid sample_strategy: {self.sample_strategy}")

    def evaluate_candidate_sequences(self, candidate_action_sequences, obs):
        # TODO(Q2): for each model in ensemble, compute the predicted sum of rewards
        # for each candidate action sequence.
        #
        # Then, return the mean predictions across all ensembles.
        # Hint: the return value should be an array of shape (N,)

        # Initializing rewards for each sequence to 0
        sum_rewards = np.zeros(candidate_action_sequences.shape[0])

        for model in self.dyn_models:
            sum_rewards += self.calculate_sum_of_rewards(obs, candidate_action_sequences, model)
            # print(f"Sum of rewards: {sum_rewards}")

        # calculate mean rewards
        mean_rewards = sum_rewards / len(self.dyn_models)
        # print(f"Mean rewards: {mean_rewards}")

        return mean_rewards

    def get_action(self, obs):
        if self.data_statistics is None:
            return self.sample_action_sequences(num_sequences=1, horizon=1)[0]

        # sample random actions (N x horizon)
        candidate_action_sequences = self.sample_action_sequences(
            num_sequences=self.N, horizon=self.horizon, obs=obs)

        if candidate_action_sequences.shape[0] == 1:
            # CEM: only a single action sequence to consider; return the first action
            return candidate_action_sequences[0][0][None]
        else:
            # print(f"Candidate {candidate_action_sequences.shape} action sequences")
            # print(f"Evaluating {candidate_action_sequences} candidate action sequences")

            predicted_rewards = self.evaluate_candidate_sequences(candidate_action_sequences, obs)
            # print(f"predicted_rewards: {predicted_rewards.shape}")

            # pick the action sequence and return the 1st element of that sequence
            best_action_sequence_index = np.argmax(predicted_rewards)
            best_action_sequence = candidate_action_sequences[best_action_sequence_index]  # TODO (Q2)
            # print(f"Best action sequence: {best_action_sequence}")
            action_to_take = best_action_sequence[0]  # TODO (Q2)
            # print(f"Action to take: {action_to_take}")

            return action_to_take[None]  # Unsqueeze the first index

    def calculate_sum_of_rewards(self, obs, candidate_action_sequences, model):
        """

        :param obs: numpy array with the current observation. Shape [D_obs]
        :param candidate_action_sequences: numpy array with the candidate action
        sequences. Shape [N, H, D_action] where
            - N is the number of action sequences considered
            - H is the horizon
            - D_action is the action of the dimension
        :param model: The current dynamics model.
        :return: numpy array with the sum of rewards for each action sequence.
        The array should have shape [N].
        """
        sum_of_rewards = None  # TODO (Q2)
        # For each candidate action sequence, predict a sequence of
        # states for each dynamics model in your ensemble.
        # Once you have a sequence of predicted states from each model in
        # your ensemble, calculate the sum of rewards for each sequence
        # using `self.env.get_reward(predicted_obs, action)` at each step.
        # You should sum across `self.horizon` time step.
        # Hint: you should use model.get_prediction and you shouldn't need
        #       to import pytorch in this file.
        # Hint: Remember that the model can process observations and actions
        #       in batch, which can be much faster than looping through each
        #       action sequence.

        # Initializing rewards for each sequence to 0
        sum_of_rewards = np.zeros(candidate_action_sequences.shape[0])

        # run the model on candidate action sequences in batch mode
        # print(f"Running model on {candidate_action_sequences.shape} candidate action sequences")
        # print(candidate_action_sequences[0])
        # print(candidate_action_sequences[0, 0, :])
        # print(candidate_action_sequences[:, 0, :].shape)

        # print(f"Predicted obs: {predicted_obs.shape}")

        obs_next = np.repeat(obs[None, :], candidate_action_sequences.shape[0], axis=0)
        # print("obs_batch: ", obs_next.shape)
        # print(obs_next)

        # obs_next = np.expand_dims(obs_batch, 0)

        # print(f"obs_next: {obs_next.shape}")

        # calculate the sum of rewards for each sequence
        for i in range(self.horizon):

            obs_next = model.get_prediction(obs_next, candidate_action_sequences[:, i, :], self.data_statistics)
            # print(f"predicted {obs_next.shape}")
            # print(f"action {candidate_action_sequences[:, i, :].shape}")

            rewards = self.env.get_reward(obs_next, candidate_action_sequences[:, i, :])

            # print(f"rewards: {rewards[0]}")

            # calculate the sum of rewards for each sequence
            sum_of_rewards += rewards[0]


        # print(f"Sum of rewards: {sum_of_rewards.shape}")


        # for i, candidate_action_sequence in enumerate(candidate_action_sequences):
        #     # Setup reward of current sequence
        #     sum_of_reward = 0

        #     obs_next = np.expand_dims(obs, 0)
        #     # print("!!!!running new sequence")
        #     # batch_obs = np.zeros((candidate_action_sequence.shape[0], obs_next.shape[1]))
        #     # batch_actions = np.zeros((candidate_action_sequence.shape[0], candidate_action_sequence.shape[2]))

        #     # For each horizon actions in the sequence, calculate the sum of rewards
        #     for candidate_action in candidate_action_sequence:
        #         action = np.expand_dims(candidate_action, 0)

        #         # Make prediction and update current obs with the prediction
        #         obs_next = model.get_prediction(obs_next, action, self.data_statistics)

        #         # Calculate reward for this action and generated observation
        #         reward = self.env.get_reward(obs_next, action)
        #         # print(f"Reward: {reward}")

        #         # Update sum of rewards
        #         sum_of_reward += reward[0][0]
        #         # print(f"Sum of reward: {sum_of_reward}")

        #     # Update sum of rewards for this sequence
        #     sum_of_rewards[i] = sum_of_reward

        # # print(f"Sum of rewards: {sum_of_rewards.shape}")


        return sum_of_rewards

