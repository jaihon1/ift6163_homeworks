#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Q2
q2_eval_avg_return = [-33.97]
q2_train_avg_return = [-167.10]
q2_itr = [1]


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
