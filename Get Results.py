# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 10:17:00 2023

@author: svalizad
"""

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt 
import os 
from scipy.stats import norm

directory = 'C:\\Users\\svalizad\\Desktop\\Automated ML\\Variables_after_code_running'

os.chdir(directory)

def normalize(data, mean, std):
    
    data = np.sort(data)
    
    normalized_data = norm.pdf(data, mean, std)
    
    """
    maximum = np.max(normalized_data) 
    minimum = np.min(normalized_data)
    
    normalized_data = (normalized_data - minimum) / (maximum - minimum)
    """
    
    return normalized_data

def get_accuracy(Rew):

    X = (3/(Rew + 1.5)) - 1    
    
    X = np.log(X)
    
    X = -1 * X
    
    X = X + 0.75
    
    return X

save_directory = directory

Best_Action = pickle.load(open(os.path.join(save_directory, 'Best_Action'), 'rb'))
Policy_Reward_History = pickle.load(open(os.path.join(save_directory, 'Policy_Reward_History'), 'rb'))
Q_Values = pickle.load(open(os.path.join(save_directory, 'Q_Values'), 'rb'))
Q_after_Iterations = pickle.load(open(os.path.join(save_directory, 'Q_after_Iterations'), 'rb'))

Q_Values_pd = pd.DataFrame(Q_after_Iterations)
Q_Values_pd.to_excel(os.path.join(save_directory, 'Q_Values_pd.xlsx'))



rewards_for_two_cases = pickle.load(open(os.path.join(save_directory, 'rewards_for_two_cases'), 'rb'))
totla_reward_for_two_cases_mean = pickle.load(open(os.path.join(save_directory, 'totla_reward_for_two_cases_mean'), 'rb'))
totla_reward_for_two_cases_std = pickle.load(open(os.path.join(save_directory, 'totla_reward_for_two_cases_std'), 'rb'))
rewards_pdf_for_two_cases = pickle.load(open(os.path.join(save_directory, 'rewards_pdf_for_two_cases'), 'rb'))

Rewards_at_each_iteration = np.mean(Policy_Reward_History, axis = 1)

mean_reward_for_each_option = np.mean(Rewards_at_each_iteration, axis = 1)

reward_variance_for_each_option = np.var(Rewards_at_each_iteration, axis = 1)

reward_variance_for_each_option = np.sqrt(reward_variance_for_each_option)

Q_after_Iterations = np.mean(Q_Values, axis = 0)


normalized_1 = normalize(Rewards_at_each_iteration[2,:], mean_reward_for_each_option[2], reward_variance_for_each_option[2])
normalized_2 = normalize(Rewards_at_each_iteration[1,:], mean_reward_for_each_option[1], reward_variance_for_each_option[1])
normalized_3 = normalize(Rewards_at_each_iteration[8,:], mean_reward_for_each_option[8], reward_variance_for_each_option[8])*0.8

plt.figure()

plt.scatter(np.sort(rewards_for_two_cases[0,:]), rewards_pdf_for_two_cases[0], label = 'Policy chosen by the machine')
plt.scatter(np.sort(rewards_for_two_cases[1,:]), rewards_pdf_for_two_cases[1], label = 'Policy chosen based on domain knowledge')
plt.plot(np.sort(rewards_for_two_cases[0,:]), rewards_pdf_for_two_cases[0])
plt.plot(np.sort(rewards_for_two_cases[1,:]), rewards_pdf_for_two_cases[1])

test_case_1 = {'First': np.sort(rewards_for_two_cases[0,:]), 'Second': rewards_pdf_for_two_cases[0]}
test_case_1_pd = pd.DataFrame(test_case_1)
test_case_1_pd.to_excel(os.path.join(save_directory, 'test_case_1_pd.xlsx'))

test_case_2 = {'First': np.sort(rewards_for_two_cases[1,:]), 'Second': rewards_pdf_for_two_cases[1]}
test_case_2_pd = pd.DataFrame(test_case_2)
test_case_2_pd.to_excel(os.path.join(save_directory, 'test_case_2_pd.xlsx'))

plt.legend()

plt.xlabel('Rewards Distribution for 100 iterations')
plt.ylabel('Normalized Gaussian function')
plt.grid()

plt.figure()

plt.plot(np.sort(Rewards_at_each_iteration[2,:]), normalized_1, label = 'First best policy (Third Policy) [Q = 0.1422]')
plt.plot(np.sort(Rewards_at_each_iteration[1,:]), normalized_2, label = 'Second best policy (Second Policy) [Q = 0.1297]')
plt.plot(np.sort(Rewards_at_each_iteration[8,:]), normalized_3, label = 'Third best policy (First Policy) [Q = 0.1198]') 

best_policy = {'First': np.sort(Rewards_at_each_iteration[2,:]), 'Second': normalized_1}
best_policy_pd = pd.DataFrame(best_policy)
best_policy_pd.to_excel(os.path.join(save_directory, 'best_policy_pd.xlsx'))

best_policy_2 = {'First': np.sort(Rewards_at_each_iteration[1,:]), 'Second': normalized_2}
best_policy_2_pd = pd.DataFrame(best_policy_2)
best_policy_2_pd.to_excel(os.path.join(save_directory, 'best_policy_2_pd.xlsx'))

best_policy_3 = {'First': np.sort(Rewards_at_each_iteration[8,:]), 'Second': normalized_3}
best_policy_3_pd = pd.DataFrame(best_policy_3)
best_policy_3_pd.to_excel(os.path.join(save_directory, 'best_policy_3_pd.xlsx'))


plt.grid()
plt.legend()

plt.xlabel('Reward Distribution over Iterations')
plt.ylabel('Distribution_function')


best_action_pd = pd.DataFrame(Best_Action)
best_action_pd.to_excel(os.path.join(save_directory, 'bestactions.xlsx'))

Q_values_pd = pd.DataFrame(Q_after_Iterations)
Q_values_pd.to_excel(os.path.join(save_directory, 'Q_values_pd.xlsx'))

action_Selected = np.zeros([60,50])

for i in range(Policy_Reward_History.shape[0]):
    rew = Policy_Reward_History[i,:,:]
    
    for j in range(Policy_Reward_History.shape[2]):
        rew_in_episode = rew[:,j]
        
        count = np.shape(np.where(rew_in_episode > 0))[1]
        
        action_Selected[i,j] = count
        
        
plt.figure()
plt.plot(np.arange(1,51), action_Selected[2,:])
plt.plot(np.arange(1,51), action_Selected[1,:])
plt.plot(np.arange(1,51), action_Selected[8,:])
        

action_Selected_pd = pd.DataFrame(action_Selected)
action_Selected_pd.to_excel(os.path.join(save_directory, 'action_Selected_pd.xlsx'))
        
    





"""

Rewards_at_each_iteration = pickle.load(open(os.path.join(save_directory, 'Rewards_at_each_iteration'), 'rb'))
mean_reward_for_each_option = pickle.load(open(os.path.join(save_directory, 'mean_reward_for_each_option'), 'rb'))
reward_std_for_each_option = pickle.load(open(os.path.join(save_directory, 'reward_std_for_each_option'), 'rb'))

"""