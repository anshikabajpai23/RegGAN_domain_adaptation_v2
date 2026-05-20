
import numpy as np, glob, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

dess = sorted(glob.glob('/N/project/prostate_cancer_ai/anshika/regGAN/preprocessed/slices/dess/*.npy'))
pd   = sorted(glob.glob('/N/project/prostate_cancer_ai/anshika/regGAN/preprocessed/slices/pd/*.npy'))

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(np.load(dess[len(dess)//2]), cmap='gray')
axes[0].set_title('DESS preprocessed')
axes[1].imshow(np.load(pd[len(pd)//2]),   cmap='gray')
axes[1].set_title('PD preprocessed')
plt.savefig('/N/project/prostate_cancer_ai/anshika/regGAN/results/debug_step1_preprocess.png', dpi=150)
print('saved')
