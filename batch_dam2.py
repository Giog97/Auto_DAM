import os
import subprocess
from PIL import Image
import random
import time

# CONFIGURAZIONE BASE
image_folder = "images_test"
output_root = "output_test"
script_path = "examples/dam_with_sam_self_contained.py"

# Prende il punto centrale dell'immagine
def get_center_point(width, height):
    return [[width // 2, height // 2]]

# Prende 4 punti centrati nell'immagine divisa in 4 parti
def get_4grid_points(width, height):
    return [
        [width // 4, height // 4],
        [3 * width // 4, height // 4],
        [width // 4, 3 * height // 4],
        [3 * width // 4, 3 * height // 4],
    ]

# Prende il bounding box centrale grande il 50% dell'immagine
def get_center_box(width, height):
    w_box = width // 2
    h_box = height // 2
    x0 = (width - w_box) // 2
    y0 = (height - h_box) // 2
    return [[x0, y0, x0 + w_box, y0 + h_box]]

# Prende i bounding box che stanno ai bordi dell'immagine
def get_corner_boxes(width, height):
    w_box = width // 4
    h_box = height // 4
    return [
        [0, 0, w_box, h_box],
        [width - w_box, 0, width, h_box],
        [0, height - h_box, w_box, height],
        [width - w_box, height - h_box, width, height],
    ]

# Prende i 4 punti ai bordi dell'immagine
def get_corner_points(width, height):
    return [
        [0, 0],
        [width - 1, 0],
        [0, height - 1],
        [width - 1, height - 1],
    ]

# Prende 9 punti disposti a griglia nell'immagine
def get_9grid_points(width, height):
    """Griglia 3x3 (9 punti)"""
    points = []
    for i in [1, 2, 3]:
        for j in [1, 2, 3]:
            x = width * i // 4
            y = height * j // 4
            points.append([x, y])
    return points

# Prende 3 box centrati e leggermente spostati casualmente nell'immagine
def get_random_centered_boxes(width, height, n=3):
    """n box centrati con leggero offset casuale (per ora deterministico)"""
    boxes = []
    for i in range(n):
        scale = 0.5 + 0.1 * i  # variazione deterministica tra 50% e 70% circa
        w_box = int(width * scale / 2)
        h_box = int(height * scale / 2)
        x0 = (width - w_box) // 2 + (i * 5)
        y0 = (height - h_box) // 2 + (i * 5)
        boxes.append([x0, y0, x0 + w_box, y0 + h_box])
    return boxes

# Prende i 4 punti agli angoli dell'immagine
def get_edge_points(width, height):
    """Punti centrati sui 4 bordi"""
    return [
        [width // 2, 0],                # Top center
        [width // 2, height - 1],       # Bottom center
        [0, height // 2],               # Left center
        [width - 1, height // 2],       # Right center
    ]

# Prende 1 punto random dell'immagine
def get_random_point(width, height):
    x = random.randint(0, width - 1)
    y = random.randint(0, height - 1)
    return [[x, y]]

# Prende 5 punti random dell'immagine
def get_random_points(width, height, n=5):
    return [[random.randint(0, width - 1), random.randint(0, height - 1)] for _ in range(n)]

# Prende una bounding box casuale grande il 30% dell'immagine
def get_random_bbox(width, height, scale=0.3):
    box_width = int(width * scale)
    box_height = int(height * scale)

    max_x = width - box_width
    max_y = height - box_height

    x_min = random.randint(0, max_x)
    y_min = random.randint(0, max_y)

    x_max = x_min + box_width
    y_max = y_min + box_height

    return [[x_min, y_min, x_max, y_max]]

# MAPPATURA MODALITÀ
modes = {
    "center_point": get_center_point,
    "4grid_points": get_4grid_points,
    "center_box": get_center_box,
    "corner_boxes": get_corner_boxes,
    "corner_points": get_corner_points,
    "9grid_points": get_9grid_points,
    "random_centered_boxes": get_random_centered_boxes,
    "edge_points": get_edge_points,
    "random_point": get_random_point,
    "random_5points": get_random_points,
    "random_box": get_random_bbox,
}

# CICLO PRINCIPALE
for mode_name, mode_func in modes.items():
    print(f"== Modalità: {mode_name} ==")
    output_folder = os.path.join(output_root, mode_name)
    os.makedirs(output_folder, exist_ok=True)

    # Inizio calcolo del tempo
    mode_start_time = time.time()
    image_count = 0

    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_path = os.path.join(image_folder, filename)
            output_image_path = os.path.join(output_folder, f"out_{filename}")

            # Conteggio immagini
            image_count += 1

            # Ottieni dimensioni
            with Image.open(image_path) as img:
                width, height = img.size

            # Genera input specifico
            input_data = mode_func(width, height)

            command = [
                "python", script_path,
                "--image_path", image_path,
                "--output_image_path", output_image_path,
            ]

            input_str = str(input_data)
            if "box" in mode_name:
                command += ["--use_box", "--box", input_str] # Puoi gestire più box se il tuo script lo supporta
            else:
                command += ["--points", input_str] # Anche qui, se serve una lista puoi modificarlo

            print(f"Modalità {mode_name} | Immagine: {filename} | Input: {input_data}")

            # Esegui comando
            result = subprocess.run(command, capture_output=True, text=True)
            stdout = result.stdout

            # Estrai descrizione
            lines = stdout.splitlines()
            description = "[Descrizione non trovata]"
            for i, line in enumerate(lines):
                if line.strip().startswith("Description:"):
                    description_lines = []
                    for line in lines[i + 1:]:
                        if line.strip().startswith("Output image"):
                            break
                        description_lines.append(line)
                    description = "\n".join(description_lines).strip()
                    break

            text_output_path = os.path.join(output_folder, f"description_{filename}.txt")
            with open(text_output_path, "w", encoding="utf-8") as f:
                f.write(description)
                
    # Fine calcolo del tempo
    mode_end_time = time.time()
    total_time = mode_end_time - mode_start_time
    avg_time = total_time / image_count if image_count > 0 else 0

    # Salva i tempi in un file
    timing_path = os.path.join(output_folder, "timing.txt")
    with open(timing_path, "w", encoding="utf-8") as f:
        f.write(f"Modalità: {mode_name}\n")
        f.write(f"Numero immagini: {image_count}\n")
        f.write(f"Tempo totale: {total_time:.2f} secondi\n")
        f.write(f"Tempo medio per immagine: {avg_time:.2f} secondi\n")

    print(f"Tempo totale per modalità '{mode_name}': {total_time:.2f} secondi")
    print(f"Tempo medio per immagine: {avg_time:.2f} secondi\n")
