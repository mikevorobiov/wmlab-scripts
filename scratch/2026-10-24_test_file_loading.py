#%%
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import labtools.data_io as dio


#%%
fpaths = dio.find_files(pattern="*.h5", match="2026-10-23")
print(fpaths)



# %%
data = dio.load_h5(fpaths[4])

# %%
img = data["light"][0, 45, 1]
img_mean = img.mean()   
img_base = img - img_mean 
plt.figure(1)
plt.imshow(img_base)
# %%
fig, ax = plt.subplots()

for i in range(img.shape[0]):
    if i % 100 == 0:
        print(i)
        ax.plot(img_base[i,:])
# %%
