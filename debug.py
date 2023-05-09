import json
import numpy as np 


ROOT="/DATATWO/users/mincut/Object-Centric-VideoAnswering/data"
mode = "train"
z=json.load(open(f"{ROOT}/features_trial/{mode}/mapping.json"))
freq = {}
for k,v in z.items():
    freq[v] = freq.get(v,0)+1
# print(freq)
for v in range(len(freq)):
    z = np.load(f"{ROOT}/features_trial/{mode}/{v}.npy")
    if z.shape[1] != 4*freq[v]:
        print(v)
#check mask
for v in range(len(freq)):
    z = np.load(f"{ROOT}/features_mask_trial/{mode}/{v}.npy")
    if z.shape[1] != 320*freq[v]:
        print(v) 
"####check sizes####"

for v in range(len(freq)):
    
    z = np.load(f"{ROOT}/features_dummy_new/{mode}/{v}.npy")



# z = np.load(f"{ROOT}/features_trial/{mode}/{65}.npy")
# print(z)
# breakpoint()

    