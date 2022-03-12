#%%
import numpy as np
import matplotlib.pyplot as plt


#%%
# Get data from csv
data_1 = np.genfromtxt('data/run-16-52-53_data_hw3_q4_b30000_r0.01_HalfCheetah-v2_10-03-2022_16-52-53-tag-Eval_AverageReturn.csv', delimiter=',' , skip_header=1)

data_2 = np.genfromtxt('data/run-16-53-02_data_hw3_q4_b30000_r0.01_rtg_HalfCheetah-v2_10-03-2022_16-53-02-tag-Eval_AverageReturn.csv', delimiter=',' , skip_header=1)

data_3 = np.genfromtxt('data/run-16-53-09_data_hw3_q4_b30000_r0.01_nnbaseline_HalfCheetah-v2_10-03-2022_16-53-09-tag-Eval_AverageReturn.csv', delimiter=',' , skip_header=1)

data_4 = np.genfromtxt('data/run-16-53-20_data_hw3_q4_b30000_r0.01_rtg_nnbaseline_HalfCheetah-v2_10-03-2022_16-53-20-tag-Eval_AverageReturn.csv', delimiter=',' , skip_header=1)

# data_5 = np.genfromtxt('data/run-16-01-27_data_hw3_q4_search_b10000_lr0.01_rtg_nnbaseline_HalfCheetah-v2_10-03-2022_16-01-27-tag-Eval_AverageReturn.csv', delimiter=',' , skip_header=1)




#%%
means_1 = []
means_2 = []
means_3 = []
means_4 = []
means_5 = []


std_1 = []
std_2 = []
std_3 = []

labels = []

# for i, mean1 in enumerate(data_1):
#     means_1.append(mean1[2])

#     labels.append(mean1[1])



for i, (mean1, mean2, mean3, mean4) in enumerate(zip(data_1, data_2, data_3, data_4)):
    means_1.append(mean1[2])
    means_2.append(mean2[2])
    means_3.append(mean3[2])
    means_4.append(mean4[2])
    # means_5.append(mean5[2])

    labels.append(mean1[1])



#%%

plt.errorbar(labels, means_1, color='b', linestyle='-')
plt.errorbar(labels, means_2, color='r', linestyle='-')
plt.errorbar(labels, means_3, color='g', linestyle='-')
plt.errorbar(labels, means_4, color='y', linestyle='-')
# plt.errorbar(labels, means_5, color='k', linestyle='-')



# Our y−axis is ”success rate” here.
plt.ylabel('Average Return', fontsize=10)
# Our x−axis is iteration number.
plt.xlabel('Iteration', fontsize=10, labelpad=0)
# Our task is called ”Awesome Robot Performance”
plt.title('Learning curves for HalfCheetah environment', fontsize=15)
plt.grid()
plt.legend(labels=['vanilla', 'rtg', 'nn_baseline', 'rtg nn_baseline'])


# %%
