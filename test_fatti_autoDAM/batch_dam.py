"""
Wrapper Python per lanciare dam_with_sam_self_contained.py, in modo da:
- Iterare su tutte le immagini in una cartella.
- Costruire e chiamare il comando per dam_with_sam_self_contained.py via subprocess.
"""

# Per esegurilo, mettere questo file nella directory principale del progetto describe-anything
# esegurilo con il comando: python batch_dam.py

import os
import subprocess
from PIL import Image  # Serve per ottenere dimensioni immagine (inizialmente erano fisse)

# CONFIGURAZIONE 
image_folder = "images_test"
output_folder = "output_test"
script_path = "examples/dam_with_sam_self_contained.py"

use_box = True  # se False, usa i punti (non necessario qui) perchè punti usati di default da 'dam_with_sam_self_contained.py'
# use_box messo per un uso fututo in caso si voglia usare punti

# Crea la cartella output se non esiste
os.makedirs(output_folder, exist_ok=True)

# Itera su tutti i file nella cartella immagini
for filename in os.listdir(image_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        image_path = os.path.join(image_folder, filename)
        output_image_path = os.path.join(output_folder, f"out_{filename}")

        # Ottiene dimensioni dell'immagine corrente che sta analizzando
        with Image.open(image_path) as img:
            width, height = img.size
        box = f"[0, 0, {width}, {height}]"  # box uguale alla dimensione immagine

        # Costruisc+e il comando
        command = [
            "python", script_path,
            "--image_path", image_path,
            "--output_image_path", output_image_path,
            "--use_box",
            "--box", box
        ]

        print(f"Eseguendo su: {filename} con box {box}")

        # Esegue e cattura output
        result = subprocess.run(command, capture_output=True, text=True)
        description = result.stdout

        # Estrai solo la descrizione
        lines = description.splitlines()
        start_idx = None

        # NOTA la descrizione la possiamo prendere dell'output ottenuto eseguendo per ogni immagine dam_with_sam_self_contained.py
        # La descrzione che ci interessa sta tra "Description:" e "Output image with contour saved as output_test\out_1.jpg"

        # Trova la riga che inizia con "Description:"
        for i, line in enumerate(lines):
            if line.strip().startswith("Description:"):
                start_idx = i + 1
                break

        # Se trovata, prendi tutte le righe successive fino a una riga vuota o alla fine
        if start_idx is not None:
            description_lines = []
            for line in lines[start_idx:]:
                if line.strip().startswith("Output image with contour saved"):
                    break
                description_lines.append(line)
            clean_description = "\n".join(description_lines).strip()
        else:
            clean_description = "[Descrizione non trovata]"

        # Salva solo la descrizione in un file txt
        text_output_path = os.path.join(output_folder, f"description_{filename}.txt")
        with open(text_output_path, "w", encoding="utf-8") as f:
            f.write(clean_description)



