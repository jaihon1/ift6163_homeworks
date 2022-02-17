#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Q2
q2_eval_avg_return = [-33.97]
q2_train_avg_return = [-167.10]
q2_itr = [1]

# q3
q3_obs = [-21.05]
q3_obs_std = [6.39]
q3_reach = [-689.24]
q3_reach_std = [0.00]
q3_chee = [42.00]
q3_chee_std = [0.00]
q3_itr = [1]


# q4
q4_reacher_horizon5 = [-702.24]
q4_reacher_horizon15 = []
q4_reacher_horizon30 = []
q4_reacher_numseq100 = []
q4_reacher_numseq1000 = []
q4_reacher_ensemble1 = []
q4_reacher_ensemble3 = []
q4_reacher_ensemble5 = []

q4_reacher_horizon5_std = [0.00]
q4_reacher_horizon15_std = [0.00]
q4_reacher_horizon30_std = [0.00]
q4_reacher_numseq100_std = [0.00]
q4_reacher_numseq1000_std = [0.00]
q4_reacher_ensemble1_std = [0.00]
q4_reacher_ensemble3_std = [0.00]
q4_reacher_ensemble5_std = [0.00]

q4_iter = [1]


#%%

sns.scatterplot(x=q2_itr, y=q2_eval_avg_return, color='r', linestyle='-')
sns.scatterplot(x=q2_itr, y=q2_train_avg_return, color='b', linestyle='-')

plt.xticks(q2_itr)


# Our y−axis is ”success rate” here.
plt.ylabel('Average Return', fontsize=10)
# Our x−axis is iteration number.
plt.xlabel('Iteration', fontsize=10, labelpad=0)
# Our task is called ”Awesome Robot Performance”
plt.title('Eval and Train performances on obstacles environemnt', fontsize=15)
plt.grid()
plt.legend(labels=['Eval', 'Train'])


# %%

# sns.lineplot(x=dagger_iteration, y=dagger_mean, color='r', linestyle='-')

plt.xticks(dagger_iteration)

plt.errorbar(dagger_iteration, dagger_mean, dagger_std, linestyle='-', marker='+')
plt.errorbar(dagger_iteration, dagger_mean_expert, dagger_std_expert, color='r', linestyle='-', marker='+')

# Our y−axis is ”success rate” here.
plt.ylabel('Average Return', fontsize=10)
# Our x−axis is iteration number.
plt.xlabel('Iteration', fontsize=10, labelpad=0)
# Our task is called ”Awesome Robot Performance”
plt.title('BC and Expert agents Performance on Ant Task', fontsize=15)
plt.grid()

plt.legend(['BC agent', 'Expert agent'])
# %%

plt.xticks(dagger_iteration)

plt.errorbar(dagger_iteration_2, dagger_mean_2, dagger_std_2, linestyle='-', marker='+')
plt.errorbar(dagger_iteration_2, dagger_mean_expert_2, dagger_std_expert_2, color='r', linestyle='-', marker='+')

# Our y−axis is ”success rate” here.
plt.ylabel('Average Return', fontsize=10)
# Our x−axis is iteration number.
plt.xlabel('Iteration', fontsize=10, labelpad=0)
# Our task is called ”Awesome Robot Performance”
plt.title('BC and Expert agents Performance on Hopper Task', fontsize=15)
plt.grid()

plt.legend(['BC agent', 'Expert agent'])
# %%
