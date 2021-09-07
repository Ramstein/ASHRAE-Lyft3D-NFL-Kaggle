import numpy as np
t1 = np.arange(1,10)[None,:]
t2 = np.arange(11,20)[None,:]
t3 = np.concatenate((t1,t2),axis=1)


t1 = t1.T
print(t3)

im = np.array(im).astype(np.float32)

