import clip
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from cc3m_llava import CC3MLLaVaDataset

# Carica CLIP e preprocessing
_, preprocess = clip.load("ViT-B/32")

# Crea dataset con immagini attive
dataset = CC3MLLaVaDataset('/andromeda/datasets', preprocess, return_image=True)

# Numero di immagini da mostrare
n_show = 5

# Estrai primi n_show elementi
samples = [dataset[i] for i in range(n_show)]

# Estrai immagini e testi
images = [s['image'] for s in samples]
captions = [s['text'] for s in samples]

# Crea griglia di immagini
grid = make_grid(images, nrow=n_show)

# Visualizza con Matplotlib
plt.figure(figsize=(20, 5))
plt.imshow(grid.permute(1, 2, 0))  # cambia dimensioni da (C, H, W) a (H, W, C)
plt.axis('off')
plt.title("Prime 5 immagini del dataset", fontsize=16)
plt.show()

# Stampa le caption
for i, cap in enumerate(captions):
    print(f"[{i}] {cap}")