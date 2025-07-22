"""
Wrapper Python per lanciare dam_with_maschere_pregenerate.py, in modo da:
- Cercare l'immagine 'original' nella cartella delle maschere.
- Iterare su tutte le maschere nella stessa cartella.
- Per ogni maschera, generare la descrizione usando dam_with_maschere_pregenerate.py.
"""
import os
import subprocess
from PIL import Image

# CONFIGURAZIONE GENERALE
base_folder = "maschere_output_panoptic"  # Cartella che contiene le sottocartelle con maschere
script_path = "dam_with_maschere_pregenerate.py"

# Itera su tutte le sottocartelle
for folder in os.listdir(base_folder):
    folder_path = os.path.join(base_folder, folder)
    if not os.path.isdir(folder_path):
        continue  # Salta se non è una cartella

    print(f"[INFO] Elaborazione cartella: {folder_path}")
    output_folder = f"output_test_{folder}"
    os.makedirs(output_folder, exist_ok=True)

    # Trova immagine 'original'
    original_image_path = None
    for filename in os.listdir(folder_path):
        if 'original' in filename.lower() and filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            original_image_path = os.path.join(folder_path, filename)
            break

    if original_image_path is None:
        print(f"[WARNING] Nessuna immagine 'original' trovata in {folder_path}. Skipping...")
        continue

    # Ottiene dimensioni immagine originale
    with Image.open(original_image_path) as img:
        width, height = img.size
    box = f"[0, 0, {width}, {height}]"

    print(f"[INFO] Immagine originale: {original_image_path}, dimensioni: {width}x{height}")

    # Itera sulle maschere nella cartella
    # Itera sulle maschere contenute all'interno della directory contenente l'immagine original (in realtà itera sui file all'interno della directory folder)
    # NBB: queste cartelle su cui itera il seguente ciclo, vengono generate dallo script 'panoptic-segment-anything_rip_mod.py'
    for mask_filename in os.listdir(folder_path):
        mask_path = os.path.join(folder_path, mask_filename)

        # Salta l'immagine originale
        if mask_path == original_image_path:
            continue

        # Considera solo file .png o .npy
        if not mask_filename.lower().endswith(('.png', '.npy')):
            continue

        output_image_path = os.path.join(output_folder, f"out_{mask_filename}.png")

        command = [
            "python", script_path,
            "--image_path", original_image_path,
            "--output_image_path", output_image_path,
            "--mask_path", mask_path
        ]

        print(f"[INFO] Eseguendo su maschera: {mask_filename}")
        result = subprocess.run(command, capture_output=True, text=True)
        description = result.stdout

        # Estrai descrizione
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

        # Salva la descrizione
        description_filename = f"description_{os.path.splitext(mask_filename)[0]}.txt"
        description_path = os.path.join(output_folder, description_filename)
        with open(description_path, "w", encoding="utf-8") as f:
            f.write(clean_description)
        print(f"[INFO] Descrizione salvata in: {description_path}")

print("[FINE] Elaborazione completata.")



