import numpy as np
import pickle
import configs
from sklearn.utils import shuffle
seq_len = configs.seq_length
seq_step = configs.seq_interval

data_set = []
data_target = []
opt_path = [1,5,6,8,9]
for i in range(1,10):
    data = []
    target = []
    path = './data/Protocol/subject10'+str(i)+'.dat'
    with open(path, 'r') as f:
        d = f.readlines()
        beat = None
        for i in d:
            k = i.rstrip().split(" ")
            seq = [float(i) for i in k]
            seq[1] = int(seq[1])

            if not np.isnan(seq[2]):
                beat = seq[2]
                if np.isnan(seq[2:]).sum()==0:
                    data.append(seq[2:]) 
                t = [0]*25
                t[seq[1]] = 1
                
                target.append(t)
            elif beat:
                seq[2] = beat 
                if np.isnan(seq[2:]).sum()==0:
                    data.append(seq[2:]) 
                t = [0]*25
                t[seq[1]] = 1
                target.append(t)
    total = (len(data)-seq_len)//seq_step
    for i in range(total):
        data_set.append(data[i*seq_step:i*seq_step + seq_len])
        data_target.append(target[i*seq_step + seq_len-1])


for i in opt_path:
    data = []
    target = []
    path = 'data/Optional/subject10'+str(i)+'.dat'
    with open(path, 'r') as f:
        d = f.readlines()
        beat = None
        for i in d:
            k = i.rstrip().split(" ")
            seq = [float(i) for i in k]
            seq[1] = int(seq[1])

            if not np.isnan(seq[2]):
                beat = seq[2]
                if np.isnan(seq[2:]).sum()==0:
                    data.append(seq[2:]) 
                t = [0]*25
                t[seq[1]] = 1
                
                target.append(t)
            elif beat:
                seq[2] = beat 
                if np.isnan(seq[2:]).sum()==0:
                    data.append(seq[2:]) 
                t = [0]*25
                t[seq[1]] = 1
                target.append(t)
    
    total = (len(data)-seq_len)//seq_step
    for i in range(total):
        data_set.append(data[i*seq_step:i*seq_step + seq_len])
        data_target.append(target[i*seq_step + seq_len-1])

data_set, data_target = shuffle(data_set, data_target, random_state=0)

with open('feature', 'wb') as fp:
    pickle.dump(data_set, fp)
fp.close()

with open('target', 'wb') as fp:
    pickle.dump(data_target, fp)
fp.close()

with open('small_sample', 'wb') as fp:
    pickle.dump(data_set[:10], fp)
fp.close()