#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Ant BC
bc_hyper_mean = [657.77, 3192.80, 3926.34, 4420.42, 4598.42, 4427.07, 4608.40, 4293.07, 4631.33, 3671.53]
bc_hyper_std = [115.31, 1544.95, 1291.36, 65.98, 92.22, 77.16, 70.32, 508.11, 85.97, 1397.67]
bc_hyper_value = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Ant Dagger
dagger_mean = [3671.53, 4649.99, 4716.66, 4760.87, 4006.90, 3955.38, 4764.33, 3992.00, 4757.17, 4893.93]
dagger_std = [1397.67, 62.03, 22.37, 111.63, 1475.82, 1664.50, 60.52, 1551.12, 120.29, 44.36]
dagger_mean_expert = [4713.65, 4438.06, 4659.47, 4727.44, 4709.90, 4567.22, 4630.74, 4678.63, 4726.93, 4748.02]
dagger_std_expert = [12.19, 41.95, 134.74, 77.85, 115.14, 207.97, 135.05, 242.77, 71.84, 122.38]
dagger_iteration = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Hopper Dagger

dagger_mean_2 = [1070.91, 2722.97, 3373.16, 3748.73, 3753.51,  3773.28, 3778.13, 3778.51, 3781.81, 3776.12]
dagger_std_2 = [22.35, 486.22, 388.67, 18.40, 39.22, 3.20, 3.17, 2.45, 1.06, 5.48]
dagger_mean_expert_2 = [3772.67, 1088.05, 3042.06, 3493.50, 3721.11, 3778.39, 3768.00, 3776.16, 3776.54, 3779.81]
dagger_std_expert_2 = [1.94, 97.58, 385.00, 36.36, 37.36, 9.64, 22.97, 4.47, 3.52, 2.34]
dagger_iteration_2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

#%%

sns.lineplot(x=bc_hyper_value, y=bc_hyper_mean, color='r', linestyle='-')

plt.xticks(bc_hyper_value)

plt.errorbar(bc_hyper_value, bc_hyper_mean, bc_hyper_std, linestyle='-', marker='^')


# Our y−axis is ”success rate” here.
plt.ylabel('Average Return', fontsize=10)
# Our x−axis is iteration number.
plt.xlabel('Data Ratio', fontsize=10, labelpad=0)
# Our task is called ”Awesome Robot Performance”
plt.title('Agent Performance on Ant Task', fontsize=15)
plt.grid()


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
