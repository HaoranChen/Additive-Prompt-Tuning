# incremental forgetting metric
# ASSUMES EQUAL TASK SIZE (e.g., classes per task)
# 
# y is num_trials x num_tasks_time x num_tasks_eval
# 
# so, y[t,i,j] would be the eval performance of the j-th task (third index) after
# training the i-th task (second index) in random seed trial t (first index). y[0,1,3] would be the eval performance of the
# fourth task after training the second task in random seed trial 1, and thus should not exist (will be ignored)
#
# note that for 1 random trial, input should be 1 x T x T size, where T is number of tasks
import numpy as np
def calc_coda_forgetting(y):
    y = np.asarray(y)
    fgt_all=[]
    index = y[0].shape[1]
    for r in range(len(y)):

        # calculate forgetting
        fgt=0
        for t in range(1,index):    # 1 - 10
            for i in range(t): # 0 - t-1
                fgt+= (y[r][t-1,i] - y[r][t,i]) * ((1)/(t))
        fgt_all.append(fgt / (len(y[r])-1))

    fgt_all = np.asarray(fgt_all)
    return np.mean(fgt_all), np.std(fgt_all)

def calc_general_forgetting(y):
    y = np.asarray(y)
    fgt_all=[]
    index = y[0].shape[1]
    for r in range(len(y)):

        # calculate forgetting
        fgt=0
        for t in range(index-1):    # 1 - 10
            fgt+= (y[r][t,t] - y[r][index-1,t])
        fgt_all.append(fgt / (len(y[r])-1))

    fgt_all = np.asarray(fgt_all)
    return np.mean(fgt_all), np.std(fgt_all)
