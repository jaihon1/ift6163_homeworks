#%%
import numpy as np
import matplotlib.pyplot as plt


#%%
# Get data from csv
data_1 = np.genfromtxt('data_plot/q1/run-14-20-55_data_hw3_q1_MsPacman-v0_01-04-2022_14-20-55-tag-Train_AverageReturn.csv', delimiter=',' , skip_header=1)
data_2 = np.genfromtxt('data_plot/q1/run-14-20-55_data_hw3_q1_MsPacman-v0_01-04-2022_14-20-55-tag-Train_BestReturn.csv', delimiter=',' , skip_header=1)


#%%
# Question 1

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
plt.ylabel('Return', fontsize=10)
# Our x−axis is iteration number.
plt.xlabel('Timestep', fontsize=10, labelpad=0)
# Our task is called ”Awesome Robot Performance”
plt.title('Question 1: basic Q-learning performance (DQN) Pacman', fontsize=15)
plt.grid()
plt.legend(labels=['Average return', 'Best return'])

#%%
# Get data from csv
data_1 = np.genfromtxt('data_plot/q1/run-17-08-21_data_hw3_q1_LunarLander-v3_24-03-2022_17-08-21-tag-Train_AverageReturn.csv', delimiter=',' , skip_header=1)
data_2 = np.genfromtxt('data_plot/q1/run-17-08-21_data_hw3_q1_LunarLander-v3_24-03-2022_17-08-21-tag-Train_BestReturn.csv', delimiter=',' , skip_header=1)


#%%
# Question 1 - lunar

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
plt.ylabel('Return', fontsize=10)
# Our x−axis is iteration number.
plt.xlabel('Timestep', fontsize=10, labelpad=0)
# Our task is called ”Awesome Robot Performance”
plt.title('Question 1: basic Q-learning performance (DQN) LunarLander', fontsize=15)
plt.grid()
plt.legend(labels=['Average return', 'Best return'])

#%%
# Get data from csv
data_1 = np.genfromtxt('data_plot/q2/run-16-08-37_data_hw3_q2_dqn_1_LunarLander-v3_24-03-2022_16-08-37-tag-Train_AverageReturn.csv', delimiter=',' , skip_header=1)
data_2 = np.genfromtxt('data_plot/q2/run-16-08-56_data_hw3_q2_dqn_1_LunarLander-v3_24-03-2022_16-08-56-tag-Train_AverageReturn.csv', delimiter=',' , skip_header=1)
data_3 = np.genfromtxt('data_plot/q2/run-16-09-11_data_hw3_q2_dqn_1_LunarLander-v3_24-03-2022_16-09-11-tag-Train_AverageReturn.csv', delimiter=',' , skip_header=1)

data_4 = np.genfromtxt('data_plot/q2/run-16-16-26_data_hw3_q2_doubledqn_1_LunarLander-v3_24-03-2022_16-16-26-tag-Train_AverageReturn.csv', delimiter=',' , skip_header=1)
data_5 = np.genfromtxt('data_plot/q2/run-16-16-37_data_hw3_q2_doubledqn_2_LunarLander-v3_24-03-2022_16-16-37-tag-Train_AverageReturn.csv', delimiter=',' , skip_header=1)
data_6 = np.genfromtxt('data_plot/q2/run-16-16-43_data_hw3_q2_doubledqn_3_LunarLander-v3_24-03-2022_16-16-43-tag-Train_AverageReturn.csv', delimiter=',' , skip_header=1)


#%%
# Question 2

means_1 = []
means_2 = []

labels = []

for i, (mean1, mean2, mean3, mean4, mean5, mean6) in enumerate(zip(data_1, data_2, data_3, data_4, data_5, data_6)):
    mean_dqn = (mean1[2] + mean2[2] + mean3[2]) / 3
    means_1.append(mean_dqn)

    mean_doubledqn = (mean4[2] + mean5[2] + mean6[2]) / 3
    means_2.append(mean_doubledqn)


    labels.append(mean1[1])


plt.errorbar(labels, means_1, color='b', linestyle='-')
plt.errorbar(labels, means_2, color='r', linestyle='-')

# Our y−axis is ”success rate” here.
plt.ylabel('Average Return', fontsize=10)
# Our x−axis is iteration number.
plt.xlabel('Timestep', fontsize=10, labelpad=0)
# Our task is called ”Awesome Robot Performance”
plt.title('Question 2: double Q-learning (DDQN)', fontsize=15)
plt.grid()
plt.legend(labels=['DQN', 'DDQN'])


#%%
# Get data from csv
data_1 = np.genfromtxt('data_plot/q3/run-14-29-24_data_hw3_q3_gamma50_LunarLander-v3_01-04-2022_14-29-24-tag-Train_AverageReturn.csv', delimiter=',' , skip_header=1)
data_2 = np.genfromtxt('data_plot/q3/run-14-29-41_data_hw3_q3_gamma90_LunarLander-v3_01-04-2022_14-29-42-tag-Train_AverageReturn.csv', delimiter=',' , skip_header=1)
data_3 = np.genfromtxt('data_plot/q3/run-14-29-50_data_hw3_q3_gamma95_LunarLander-v3_01-04-2022_14-29-50-tag-Train_AverageReturn.csv', delimiter=',' , skip_header=1)
data_4 = np.genfromtxt('data_plot/q3/run-14-29-58_data_hw3_q3_gamma99_LunarLander-v3_01-04-2022_14-29-58-tag-Train_AverageReturn.csv', delimiter=',' , skip_header=1)



#%%
# Question 3 - gamma

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
plt.xlabel('Timestep', fontsize=10, labelpad=0)
# Our task is called ”Awesome Robot Performance”
plt.title('Question 3: experimenting with hyperparameters - Gamma', fontsize=15)
plt.grid()
plt.legend(labels=['gamma=0.50', 'gamma=0.90', 'gamma=0.95', 'gamma=0.99'])


#%%
# Get data from csv
data_1 = np.genfromtxt('data_plot/q3/run-15-54-55_data_hw3_q3_lr1e-1_LunarLander-v3_01-04-2022_15-54-55-tag-Train_AverageReturn.csv', delimiter=',' , skip_header=1)
data_2 = np.genfromtxt('data_plot/q3/run-15-55-04_data_hw3_q3_lr1e-2_LunarLander-v3_01-04-2022_15-55-04-tag-Train_AverageReturn.csv', delimiter=',' , skip_header=1)
data_3 = np.genfromtxt('data_plot/q3/run-15-55-13_data_hw3_q3_lr1e-3_LunarLander-v3_01-04-2022_15-55-14-tag-Train_AverageReturn.csv', delimiter=',' , skip_header=1)
data_4 = np.genfromtxt('data_plot/q3/run-15-55-21_data_hw3_q3_lr1e-4_LunarLander-v3_01-04-2022_15-55-21-tag-Train_AverageReturn.csv', delimiter=',' , skip_header=1)



#%%
# Question 3 - learning rate

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
plt.xlabel('Timestep', fontsize=10, labelpad=0)
# Our task is called ”Awesome Robot Performance”
plt.title('Question 3: experimenting with hyperparameters - Learning Rate', fontsize=15)
plt.grid()
plt.legend(labels=['lr=1e-1', 'lr=1e-2', 'lr=1e-3', 'lr=1e-4'])



# %%
#%%
# Get data from csv
data_1 = np.genfromtxt('data_plot/q4/run-02-59-33_data_hw3_q4_ddpg_up_lr1e-1_InvertedPendulum-v2_27-03-2022_02-59-33-tag-Eval_AverageReturn.csv', delimiter=',' , skip_header=1)
data_2 = np.genfromtxt('data_plot/q4/run-02-59-44_data_hw3_q4_ddpg_up_lr1e-2_InvertedPendulum-v2_27-03-2022_02-59-45-tag-Eval_AverageReturn.csv', delimiter=',' , skip_header=1)
data_3 = np.genfromtxt('data_plot/q4/run-02-53-39_data_hw3_q4_ddpg_up_lr1e-3_InvertedPendulum-v2_27-03-2022_02-53-39-tag-Eval_AverageReturn.csv', delimiter=',' , skip_header=1)
data_4 = np.genfromtxt('data_plot/q4/run-02-59-56_data_hw3_q4_ddpg_up_lr1e-4_InvertedPendulum-v2_27-03-2022_02-59-56-tag-Eval_AverageReturn.csv', delimiter=',' , skip_header=1)



#%%
# Question 4 - learning rate

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


plt.errorbar(labels[:20], means_1[:20], color='b', linestyle='-')
plt.errorbar(labels[:20], means_2[:20], color='r', linestyle='-')
plt.errorbar(labels[:20], means_3[:20], color='g', linestyle='-')
plt.errorbar(labels[:20], means_4[:20], color='y', linestyle='-')

# Our y−axis is ”success rate” here.
plt.ylabel('Average Return', fontsize=10)
# Our x−axis is iteration number.
plt.xlabel('Timestep', fontsize=10, labelpad=0)
# Our task is called ”Awesome Robot Performance”
plt.title('Question 4: Experiments (DDPG) - Learning Rate', fontsize=15)
plt.grid()
plt.legend(labels=['lr=1e-1', 'lr=1e-2', 'lr=1e-3', 'lr=1e-4'])


# %%
#%%
# Get data from csv
data_1 = np.genfromtxt('data_plot/q4/run-03-13-15_data_hw3_q4_ddpg_up_tuf1_InvertedPendulum-v2_27-03-2022_03-13-15-tag-Eval_AverageReturn.csv', delimiter=',' , skip_header=1)
data_2 = np.genfromtxt('data_plot/q4/run-03-13-31_data_hw3_q4_ddpg_up_tuf2_InvertedPendulum-v2_27-03-2022_03-13-31-tag-Eval_AverageReturn.csv', delimiter=',' , skip_header=1)
data_3 = np.genfromtxt('data_plot/q4/run-03-13-41_data_hw3_q4_ddpg_up_tuf4_InvertedPendulum-v2_27-03-2022_03-13-41-tag-Eval_AverageReturn.csv', delimiter=',' , skip_header=1)
data_4 = np.genfromtxt('data_plot/q4/run-03-13-56_data_hw3_q4_ddpg_up_tuf8_InvertedPendulum-v2_27-03-2022_03-13-56-tag-Eval_AverageReturn.csv', delimiter=',' , skip_header=1)



#%%
# Question 4 - tuf

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


plt.errorbar(labels[:20], means_1[:20], color='b', linestyle='-')
plt.errorbar(labels[:20], means_2[:20], color='r', linestyle='-')
plt.errorbar(labels[:20], means_3[:20], color='g', linestyle='-')
plt.errorbar(labels[:20], means_4[:20], color='y', linestyle='-')

# Our y−axis is ”success rate” here.
plt.ylabel('Average Return', fontsize=10)
# Our x−axis is iteration number.
plt.xlabel('Timestep', fontsize=10, labelpad=0)
# Our task is called ”Awesome Robot Performance”
plt.title('Question 4: Experiments (DDPG) - Learning Frequency', fontsize=15)
plt.grid()
plt.legend(labels=['learning_freq=1', 'learning_freq=2', 'learning_freq=4', 'learning_freq=8'])


#%%
# Get data from csv
data_1 = np.genfromtxt('data_plot/q5/run-04-31-50_data_hw3_q5_ddpg_hard_uplf2_lr1e-2_HalfCheetah-v2_27-03-2022_04-31-50-tag-Eval_AverageReturn.csv', delimiter=',' , skip_header=1)
data_2 = np.genfromtxt('data_plot/q5/run-04-31-50_data_hw3_q5_ddpg_hard_uplf2_lr1e-2_HalfCheetah-v2_27-03-2022_04-31-50-tag-Eval_MaxReturn.csv', delimiter=',' , skip_header=1)


#%%
# Question 5

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
plt.ylabel('Return', fontsize=10)
# Our x−axis is iteration number.
plt.xlabel('Timestep', fontsize=10, labelpad=0)
# Our task is called ”Awesome Robot Performance”
plt.title('Question 5: Best parameters on a more difficult task', fontsize=15)
plt.grid()
plt.legend(labels=['Average return', 'Best return'])

#%%
# Get data from csv
data_1 = np.genfromtxt('data_plot/q6/run-17-05-53_data_hw3_q6_td3_rho0.1_InvertedPendulum-v2_29-03-2022_17-05-53-tag-Eval_AverageReturn.csv', delimiter=',' , skip_header=1)
data_2 = np.genfromtxt('data_plot/q6/run-17-06-04_data_hw3_q6_td3_rho0.2_InvertedPendulum-v2_29-03-2022_17-06-04-tag-Eval_AverageReturn.csv', delimiter=',' , skip_header=1)
data_3 = np.genfromtxt('data_plot/q6/run-17-06-13_data_hw3_q6_td3_rho0.4_InvertedPendulum-v2_29-03-2022_17-06-13-tag-Eval_AverageReturn.csv', delimiter=',' , skip_header=1)
data_4 = np.genfromtxt('data_plot/q6/run-17-06-23_data_hw3_q6_td3_rho0.8_InvertedPendulum-v2_29-03-2022_17-06-23-tag-Eval_AverageReturn.csv', delimiter=',' , skip_header=1)



#%%
# Question 6 - rho

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


plt.errorbar(labels[:20], means_1[:20], color='b', linestyle='-')
plt.errorbar(labels[:20], means_2[:20], color='r', linestyle='-')
plt.errorbar(labels[:20], means_3[:20], color='g', linestyle='-')
plt.errorbar(labels[:20], means_4[:20], color='y', linestyle='-')

# Our y−axis is ”success rate” here.
plt.ylabel('Average Return', fontsize=10)
# Our x−axis is iteration number.
plt.xlabel('Timestep', fontsize=10, labelpad=0)
# Our task is called ”Awesome Robot Performance”
plt.title('Question 6: TD3 tuning - RHO', fontsize=15)
plt.grid()
plt.legend(labels=['rho=0.1', 'rho=0.2', 'rho=0.4', 'rho=0.8'], loc='upper right')


#%%
# Get data from csv
data_1 = np.genfromtxt('data_plot/q6/run-17-30-38_data_hw3_q6_td3_tuf1_InvertedPendulum-v2_29-03-2022_17-30-38-tag-Eval_AverageReturn.csv', delimiter=',' , skip_header=1)
data_2 = np.genfromtxt('data_plot/q6/run-17-30-45_data_hw3_q6_td3_tuf2_InvertedPendulum-v2_29-03-2022_17-30-45-tag-Eval_AverageReturn.csv', delimiter=',' , skip_header=1)
data_3 = np.genfromtxt('data_plot/q6/run-17-30-53_data_hw3_q6_td3_tuf5_InvertedPendulum-v2_29-03-2022_17-30-54-tag-Eval_AverageReturn.csv', delimiter=',' , skip_header=1)
data_4 = np.genfromtxt('data_plot/q6/run-17-31-01_data_hw3_q6_td3_tuf10_InvertedPendulum-v2_29-03-2022_17-31-01-tag-Eval_AverageReturn.csv', delimiter=',' , skip_header=1)



#%%
# Question 6 - tuf

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


plt.errorbar(labels[:20], means_1[:20], color='b', linestyle='-')
plt.errorbar(labels[:20], means_2[:20], color='r', linestyle='-')
plt.errorbar(labels[:20], means_3[:20], color='g', linestyle='-')
plt.errorbar(labels[:20], means_4[:20], color='y', linestyle='-')

# Our y−axis is ”success rate” here.
plt.ylabel('Average Return', fontsize=10)
# Our x−axis is iteration number.
plt.xlabel('Timestep', fontsize=10, labelpad=0)
# Our task is called ”Awesome Robot Performance”
plt.title('Question 6: TD3 tuning - Update Frequency', fontsize=15)
plt.grid()
plt.legend(labels=['update_frequency=1', 'update_frequency=2', 'update_frequency=5', 'update_frequency=10'], loc='lower right')


#%%
# Get data from csv
data_1 = np.genfromtxt('data_plot/q7/run-16-00-04_data_hw3_q7_td3_tuf10_rho0.2_HalfCheetah-v2_01-04-2022_16-00-04-tag-Eval_AverageReturn.csv', delimiter=',' , skip_header=1)
# data_2 = np.genfromtxt('data_plot/q7/run-16-00-04_data_hw3_q7_td3_tuf10_rho0.2_HalfCheetah-v2_01-04-2022_16-00-04-tag-Eval_MaxReturn.csv', delimiter=',' , skip_header=1)


#%%
# Question 7

means_1 = []
# means_2 = []

labels = []

for i, (mean1, mean2) in enumerate(zip(data_1, data_2)):
    means_1.append(mean1[2])
    # means_2.append(mean2[2])

    labels.append(mean1[1])


plt.errorbar(labels, means_1, color='b', linestyle='-')
# plt.errorbar(labels, means_2, color='r', linestyle='-')

# Our y−axis is ”success rate” here.
plt.ylabel('Return', fontsize=10)
# Our x−axis is iteration number.
plt.xlabel('Timestep', fontsize=10, labelpad=0)
# Our task is called ”Awesome Robot Performance”
plt.title('Question 7: Evaluate TD3', fontsize=15)
plt.grid()
plt.legend(labels=['TD3'])


# %%
