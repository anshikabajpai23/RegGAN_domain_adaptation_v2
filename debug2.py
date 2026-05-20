
import numpy as np, torch, glob, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from models import Generator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
G = Generator(in_ch=1, out_ch=1, ngf=48, n_res=9).to(device)
ckpt = torch.load('/N/project/prostate_cancer_ai/anshika/regGAN/runs/run_002/ckpt_latest.pt',
                  map_location=device, weights_only=False)
G.load_state_dict(ckpt['G_AB'])
G.eval()

# load one real preprocessed slice
dess = sorted(glob.glob('/N/project/prostate_cancer_ai/anshika/regGAN/preprocessed/slices/dess/*.npy'))
sl   = np.load(dess[len(dess)//2]).astype(np.float32)

with torch.no_grad():
    t   = torch.from_numpy(sl[None, None]).to(device)
    t   = t * 2 - 1
    out = G(t)
    out = ((out + 1) / 2).squeeze().cpu().numpy()

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(sl,  cmap='gray'); axes[0].set_title('DESS input')
axes[1].imshow(out, cmap='gray'); axes[1].set_title('G_AB output (fake PD)')
plt.savefig('/N/project/prostate_cancer_ai/anshika/regGAN/results/debug_step2_generator.png', dpi=150)
print('saved  input min/max:', sl.min(), sl.max(),
      ' output min/max:', out.min(), out.max())
