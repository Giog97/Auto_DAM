"""
Wrapper Python per lanciare dam_with_maschere_pregenerate.py, in modo da:
- Cercare l'immagine 'original' nella cartella delle maschere.
- Iterare su tutte le maschere nella stessa cartella.
- Per ogni maschera, generare la descrizione usando dam_with_maschere_pregenerate.py.
"""

import os
import subprocess
from PIL import Image

# CONFIGURAZIONE
folder = "GCC_train_002582585.jpg_maschere_output_panoptic"
output_folder = "output_test_GCC_train_002582585.jpg_maschere_output_panoptic"
script_path = "dam_with_maschere_pregenerate.py"

# Crea la cartella output se non esiste
os.makedirs(output_folder, exist_ok=True)

# Trova l'immagine 'original'
original_image_path = None
for filename in os.listdir(folder):
    if 'original' in filename.lower() and filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        original_image_path = os.path.join(folder, filename)
        break

if original_image_path is None:
    raise FileNotFoundError("Nessuna immagine contenente 'original' trovata nella cartella.")

# Ottiene dimensioni dell'immagine originale
with Image.open(original_image_path) as img:
    width, height = img.size
box = f"[0, 0, {width}, {height}]"

print(f"[INFO] Immagine originale trovata: {original_image_path}")
print(f"[INFO] Dimensioni: {width}x{height}")

# Itera su tutte le maschere nella cartella
descriptions = []

# Itera sulle maschere contenute all'interno della directory contenente l'immagine original (in realtà itera sui file all'interno della directory folder)
# NBB: queste cartelle su cui itera il seguente ciclo, vengono generate dallo script 'panoptic-segment-anything_rip_mod.py'
for mask_filename in os.listdir(folder):
    mask_path = os.path.join(folder, mask_filename)
    
    # Salta il file dell'immagine originale stessa
    if mask_path == original_image_path:
        continue

    # Considera solo maschere PNG o NPY
    if not mask_filename.lower().endswith(('.png', '.npy')):
        continue

    output_image_path = os.path.join(output_folder, f"out_{mask_filename}.png")

    # Costruisci il comando base
    command = [
        "python", script_path,
        "--image_path", original_image_path,
        "--output_image_path", output_image_path,
        "--mask_path", mask_path
    ]

    print(f"[INFO] Eseguendo su maschera: {mask_filename}")

    # Esegue e cattura output
    result = subprocess.run(command, capture_output=True, text=True)
    description = result.stdout

    # Estrai la descrizione
    lines = description.splitlines()
    start_idx = None

    for i, line in enumerate(lines):
        if line.strip().startswith("Description:"):
            start_idx = i + 1
            break

    if start_idx is not None:
        description_lines = []
        for line in lines[start_idx:]:
            if line.strip().startswith("Output image with contour saved"):
                break
            description_lines.append(line)
        clean_description = "\n".join(description_lines).strip()
    else:
        clean_description = "[Descrizione non trovata]"

    # Salva la descrizione in un file txt separato
    description_filename = f"description_{os.path.splitext(mask_filename)[0]}.txt"
    description_path = os.path.join(output_folder, description_filename)
    with open(description_path, "w", encoding="utf-8") as f:
        f.write(clean_description)
    print(f"[INFO] Descrizione salvata in: {description_path}")

print(f"End")



