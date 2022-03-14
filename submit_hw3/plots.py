#%%
import numpy as np
import matplotlib.pyplot as plt


#%%
# Get data from csv
data_1 = np.genfromtxt('data/run-21-07-55_data_hw3_q7_1_100_InvertedPendulum-v2_10-03-2022_21-07-55-tag-Eval_AverageReturn.csv', delimiter=',' , skip_header=1)

data_2 = np.genfromtxt('data/run-21-08-07_data_hw3_q7_1_100_HalfCheetah-v2_10-03-2022_21-08-07-tag-Eval_AverageReturn.csv', delimiter=',' , skip_header=1)

# data_3 = np.genfromtxt('data/run-20-10-35_data_hw3_q6_1_100_CartPole-v0_10-03-2022_20-10-35-tag-Eval_AverageReturn.csv', delimiter=',' , skip_header=1)

# data_4 = np.genfromtxt('data/run-20-10-48_data_hw3_q6_10_10_CartPole-v0_10-03-2022_20-10-48-tag-Eval_AverageReturn.csv', delimiter=',' , skip_header=1)

# data_5 = np.genfromtxt('data/run-16-01-27_data_hw3_q4_search_b10000_lr0.01_rtg_nnbaseline_HalfCheetah-v2_10-03-2022_16-01-27-tag-Eval_AverageReturn.csv', delimiter=',' , skip_header=1)


#%%
# Experiment 8

means_1 = []
means_2 = []

labels = []

for i, (mean1, mean2) in enumerate(zip(data_1, data_2)):
    means_1.append(mean1[2])
    means_2.append(mean2[2])

    labels.append(mean1[1])


plt.errorbar(labels, means_1, color='b', linestyle='-')
plt.errorbar(labels, means_2, color='r', linestyle='-')

# Our y−axis is ”success rate” here.
plt.ylabel('Average Return', fontsize=10)
# Our x−axis is iteration number.
plt.xlabel('Iteration', fontsize=10, labelpad=0)
# Our task is called ”Awesome Robot Performance”
plt.title('Learning curves for Dyna', fontsize=15)
plt.grid()
plt.legend(labels=['batch_size=5000', 'batch_size=2000'])


#%%
# Experiment 7

means_1 = []
means_2 = []

labels = []

for i, mean1 in enumerate(data_2):
    if i < len(data_1):
        means_1.append(data_1[i][2])
    else:
        means_1.append(None)

    means_2.append(data_2[i][2])

    labels.append(data_2[i][1])



plt.errorbar(labels, means_1, color='b', linestyle='-')
plt.errorbar(labels, means_2, color='r', linestyle='-')

# Our y−axis is ”success rate” here.
plt.ylabel('Average Return', fontsize=10)
# Our x−axis is iteration number.
plt.xlabel('Iteration', fontsize=10, labelpad=0)
# Our task is called ”Awesome Robot Performance”
plt.title('Learning curves for Actor Critic: Complex Tasks', fontsize=15)
plt.grid()
plt.legend(labels=['InvertedPendulum-v2', 'HalfCheetah-v2'])



#%%
# Experiment 6

means_1 = []
means_2 = []
means_3 = []
means_4 = []

labels = []

for i, (mean1, mean2, mean3, mean4) in enumerate(zip(data_1, data_2, data_3, data_4)):
    means_1.append(mean1[2])
    means_2.append(mean2[2])
    means_3.append(mean3[2])
    means_4.append(mean4[2])

    labels.append(mean1[1])


plt.errorbar(labels, means_1, color='b', linestyle='-')
plt.errorbar(labels, means_2, color='r', linestyle='-')
plt.errorbar(labels, means_3, color='g', linestyle='-')
plt.errorbar(labels, means_4, color='y', linestyle='-')


# Our y−axis is ”success rate” here.
plt.ylabel('Average Return', fontsize=10)
# Our x−axis is iteration number.
plt.xlabel('Iteration', fontsize=10, labelpad=0)
# Our task is called ”Awesome Robot Performance”
plt.title('Learning curves for Actor Critic', fontsize=15)
plt.grid()
plt.legend(labels=['ntu=1 ngsptu=1', 'ntu=100 ngsptu=1', 'ntu=1 ngsptu=100', 'ntu=10 ngsptu=10'])



#%%
# Experiment 5

means_1 = []
means_2 = []
means_3 = []
means_4 = []

labels = []

for i, (mean1, mean2, mean3, mean4) in enumerate(zip(data_1, data_2, data_3, data_4)):
    means_1.append(mean1[2])
    means_2.append(mean2[2])
    means_3.append(mean3[2])
    means_4.append(mean4[2])

    labels.append(mean1[1])


plt.errorbar(labels, means_1, color='b', linestyle='-')
plt.errorbar(labels, means_2, color='r', linestyle='-')
plt.errorbar(labels, means_3, color='g', linestyle='-')
plt.errorbar(labels, means_4, color='y', linestyle='-')


# Our y−axis is ”success rate” here.
plt.ylabel('Average Return', fontsize=10)
# Our x−axis is iteration number.
plt.xlabel('Iteration', fontsize=10, labelpad=0)
# Our task is called ”Awesome Robot Performance”
plt.title('Learning curves for Generalized Advantage Estimation', fontsize=15)
plt.grid()
plt.legend(labels=['lambda=0.00', 'lambda=0.95', 'lambda=0.99', 'lambda=01.00'])




# %%
