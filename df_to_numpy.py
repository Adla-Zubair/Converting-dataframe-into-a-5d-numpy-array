# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 13:51:30 2023

@author: WS3
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 11:35:03 2023

@author: adla

converting dataframe into 3D numpy array


Need dataset in the form of (Group, Stages, Subjects)
"""

## 
conda activate mne

#%% Importing modules
import glob
import mne
import yasa
import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt


#%% Loading the dataset

result = pd.read_csv(r'features_compilation.csv')

typelist = pd.read_csv(r'metadata.csv')



#Removing unwnated columns
df = result.drop(['epochno', 'Unnamed: 0'], axis = 1)





#%% per subject per stage per electrode dataset ::  avg data

# Dataframe of unique variables
subjlist = df['subjname'].unique()
chanlist = df['Channel'].unique()
stages = df['Stage'].unique()



# Averaging the dataset for each stage per channel per subject

# Creating an empty dataframe to save the data
avg_chan = pd.DataFrame()

# Running the loop
for subj in df['subjname'].unique():
    # Loading single subject data 
    data_for_subj = df[df['subjname']== subj]
    print(subj)
    for stage in stages:
        # Loading single stage data
        print(stage)
        x = data_for_subj[data_for_subj['Stage'] ==  stage]
        for chan in chanlist:
            # Loading single channel data
            x_stage = x[x['Channel'] == chan]
            # Dropping unwanted columns
            x_stage = x_stage.drop(['subjname','Channel'], axis = 1)
            # Averaging the data
            avg_mean = x_stage.mean(axis = 0)
            # Adding info
            avg_mean['Channel'] = chan
            avg_mean['Stage'] = stage
            avg_mean['subjname'] = subj
            # Appending into a mastersheet
            avg_chan = avg_chan._append(avg_mean, ignore_index= True)

   


# Adding the CNT/MED labels to the dataset
avg_chan['Group'] = avg_chan['subjname'].map(typelist.set_index('subjname')['MED/CNT']).tolist()

# Saving the dataset
avg_chan.to_csv(r"/serverdata/ccshome/adla/NAS/DreamRecallEEG/testing_sleep_data/data/avg_chan_for_array.csv")

#%% Trial : put into loop for just perm entropy !!!!!!!!!!!!!!!!


##   Here we are making 3d array for each electrode of a particular stage for a particular group. This is then done for all stages, giving us a 4d array.
##   The 4d array of cnt and then med is then combined into a 5d array !!




# Loading the dataset for permutation entropy
perm_data = avg_chan[['perm_entropy','Channel', 'Stage','subjname','Group']]


# Loading the dataset of CNT 
data_cnt = perm_data[perm_data['Group'] ==  'CNT']
  
# Loading the dataset for each stage  -- manually enter
stage_data_cnt = data_cnt[data_cnt['Stage'] == 4]  # 0,1,2,3,4

# Creating empty list for storing the per channel data -- manually enter
cnt_data_r =[]                                     #n1,n1,n3,r     # e.g : cnt_data_r,cnt_data_w,cnt_data_n1

# Looping for each channel
for chan in chanlist:
    chan_data_cnt = stage_data_cnt[stage_data_cnt['Channel'] ==  chan]
    chan_data_cnt = chan_data_cnt.drop(['Channel','Stage', 'subjname','Group'], axis=1)
    cnt_data_r.append(chan_data_cnt)               #n1,n1,n3,r  # e.g : cnt_data_r,cnt_data_w,cnt_data_n1
    
    
#  3d array for (#elctrode, #subject, #perm values)   
stack_perm_cnt_r = np.stack(cnt_data_r, axis = 0)   
stack_perm_cnt_w = np.stack(cnt_data_w, axis = 0)   
stack_perm_cnt_n1 = np.stack(cnt_data_n1, axis = 0)   
stack_perm_cnt_n2 = np.stack(cnt_data_n2, axis = 0)   
stack_perm_cnt_n3 = np.stack(cnt_data_n3, axis = 0)   
   
    
# 4d array for (#stage, #elctrode, #subject, #perm values)    
stack_perm_cnt = np.stack((stack_perm_cnt_r,stack_perm_cnt_w,stack_perm_cnt_n1,stack_perm_cnt_n2,stack_perm_cnt_n3),axis=0)
    
    
# Loading the dataset of MED
data_med = perm_data[perm_data['Group'] ==  'MED']
   
# Loading the dataset for each stage  -- manually enter 
stage_data_med = data_med[data_med['Stage'] == 0]   # 0,1,2,3,4

# Creating empty list for storing the per channel data -- manually enter
med_data_w =[]                                        #n1,n1,n3,r     # e.g : med_data_r,med_data_w,med_data_n1

# Looping for each channel
for chan in chanlist:
    chan_data_med = stage_data_med[stage_data_med['Channel'] ==  chan]
    chan_data_med = chan_data_med.drop(['Channel','Stage', 'subjname','Group'], axis=1)
    med_data_w.append(chan_data_cnt)               #n1,n1,n3,r  # e.g : med_data_r,med_data_w,med_data_n1
    
    
#  3d array for (#elctrode, #subject, #perm values)     
stack_perm_med_r = np.stack(med_data_r, axis = 0)   
stack_perm_med_w = np.stack(med_data_w, axis = 0)   
stack_perm_med_n1 = np.stack(med_data_n1, axis = 0)   
stack_perm_med_n2 = np.stack(med_data_n2, axis = 0)   
stack_perm_med_n3 = np.stack(med_data_n3, axis = 0)   
   
    
# 4d array for (#stage, #elctrode, #subject, #perm values)     
stack_perm_med = np.stack((stack_perm_med_r,stack_perm_med_w,stack_perm_med_n1,stack_perm_med_n2,stack_perm_med_n3),axis=0)
    
# Final 5d array !!
stack_perm = np.stack((stack_perm_cnt,stack_perm_med), axis = 0)
     
    
#%% looping for all features  
    
    
##   Here we are making 3d array for each electrode of a particular stage for a particular group. This is then done for all stages, giving us a 4d array.
##   The 4d array of cnt and then med is then combined into a 5d array !!


# Loading the dataset of CNT
data_cnt = avg_chan[avg_chan['Group'] ==  'CNT']
    
# Loading the dataset for each stage  -- manually enter    
stage_data_cnt = data_cnt[data_cnt['Stage'] == 4]   # 0,1,2,3,4

# Creating empty list for storing the per channel data -- manually enter
cnt_data_r =[]                                     #n1,n1,n3,r     # e.g : cnt_data_r,cnt_data_w,cnt_data_n1

# Looping for each channel
for chan in chanlist:
    chan_data_cnt = stage_data_cnt[stage_data_cnt['Channel'] ==  chan]
    chan_data_cnt = chan_data_cnt.drop(['Channel','Stage', 'subjname','Group'], axis=1)
    cnt_data_r.append(chan_data_cnt)               #n1,n1,n3,r  # e.g : cnt_data_r,cnt_data_w,cnt_data_n1
    
    
#  3d array for (#elctrode, #subject, #values)      
stack_cnt_r = np.stack(cnt_data_r, axis = 0)   
stack_cnt_w = np.stack(cnt_data_w, axis = 0)   
stack_cnt_n1 = np.stack(cnt_data_n1, axis = 0)   
stack_cnt_n2 = np.stack(cnt_data_n2, axis = 0)   
stack_cnt_n3 = np.stack(cnt_data_n3, axis = 0)   
   
    
# 4d array for (#stage, #elctrode, #subject, #values) 
stack_cnt = np.stack((stack_cnt_w,stack_cnt_n1,stack_cnt_n2,stack_cnt_r,stack_cnt_n3),axis=0)
    
    
# Loading the dataset of MED       
data_med = avg_chan[avg_chan['Group'] ==  'MED']
     
# Loading the dataset for each stage  -- manually enter  
stage_data_med = data_med[data_med['Stage'] == 4]   # 0,1,2,3,4

# Creating empty list for storing the per channel data -- manually enter
med_data_r =[]                                     #n1,n1,n3,r     # e.g : med_data_r,med_data_w,med_data_n1

# Looping for each channel
for chan in chanlist:
    chan_data_med = stage_data_med[stage_data_med['Channel'] ==  chan]
    chan_data_med = chan_data_med.drop(['Channel','Stage', 'subjname','Group'], axis=1)
    med_data_r.append(chan_data_med)               #n1,n1,n3,r     # e.g : med_data_r,med_data_w,med_data_n1
    
    
#  3d array for (#elctrode, #subject, #values)     
stack_med_r = np.stack(med_data_r, axis = 0)   
stack_med_w = np.stack(med_data_w, axis = 0)   
stack_med_n1 = np.stack(med_data_n1, axis = 0)   
stack_med_n2 = np.stack(med_data_n2, axis = 0)   
stack_med_n3 = np.stack(med_data_n3, axis = 0)   
   
    
# 4d array for (#stage, #elctrode, #subject, #values)     
stack_med = np.stack((stack_med_w,stack_med_n1,stack_med_n2,stack_med_r,stack_med_n3),axis=0)
    


##  stack cnt is of (5,21,21,20) and stack med is of (5,21,24,20). Make them equal for stacking by changing the stack cnt dimension to that of stack med

# Creating nan filled nump array 
def nans(shape, dtype=float):
    a = np.empty(shape, dtype)
    a.fill(np.nan)
    return a

dummy_array = nans([5,21,3,20])

# Dummy array added stack cnt to mak it ofequal dimension to stack med
stack_cnt_trial = np.concatenate((stack_cnt,dummy_array), axis = 2)

# Stacking for 5d array (#group, #stage, #elctrode, #subject, #values)
stack_all = np.stack((stack_med,stack_cnt_trial), axis = 0)  
   
 
#%%   sanity check

# Subjlist of CNT and MED in order
subjlist_cnt = data_cnt['subjname'].unique()
subjlist_med = data_med['subjname'].unique()

#1
avg_chan.loc[106]   
stack_all[1,0,1,0,]
#2
avg_chan.loc[58] 
stack_all[0,2,16,0,]
#3
avg_chan.loc[588]
stack_all[1,3,0,2,]    
#4
avg_chan.loc[3889]
stack_all[0,0,4,19,]
#5    
avg_chan.loc[1289] 
stack_all[1,1,8,6,]   
#6    
avg_chan.loc[189]    
stack_all[1,4,0,0,]    
#7
avg_chan.loc[3989]    
stack_all[0,4,20,19,]
#8
avg_chan.loc[789]
stack_all[0,2,12,3,]
#9
avg_chan.loc[9]
stack_all[0,0,9,0,]
#10
avg_chan.loc[4724]
stack_all[0,4,20,23,]
#11
avg_chan.loc[4500]
stack_all[1,4,6,20,]

#%% saving as matlab file

import scipy.io
scipy.io.savemat('stacked_array_medcnt.mat', {'stack_all' : stack_all})
