import os
import time
import random
import numpy as np
import torch
from PIL import Image
from transformers import SamModel, SamProcessor, AutoModel
import cv2
from typing import List, Tuple
"""
In questo script sono definite la classe DAMProcessor che serve per inizializzare DAM e SAM per ottenere caption dalle immagini.
Inoltre sono presenti:
- generate_grid_points: utili per generare le griglie di punti per livello.
- process_single_image: serve per processare una immagine in base al suo path
- process_image_folder: serve per processare tutte le immagini in una directory di un path 
"""

class DAMProcessor:
    """
    Classe che serve per inzializzare DAM e SAM model utili per ottenere caption dalle immagini
    """
    # Define the presets at class level. Class variable that permit to reproduce experiment obtaining different caption 
    PRESET_PARAMS = {
        'conservative': {
            'temperature': 0.2, #0.2, 0.5
            'top_p': 0.5, #0.5, 0.8
            'max_new_tokens': 512
        },
        'balanced': {
            'temperature': 0.6, #0.6, 0.75
            'top_p': 0.75, #0.75, 0.9
            'max_new_tokens': 512
        },
        'creative': {
            'temperature': 1.0,
            'top_p': 0.95,
            'max_new_tokens': 512
        }
    }

    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize SAM and DAM models"""
        print("Initializing SAM model...")
        self.sam_model = SamModel.from_pretrained("facebook/sam-vit-huge").to(self.device)
        self.sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
        
        print("Initializing DAM model...")
        self.dam_model = AutoModel.from_pretrained(
            'nvidia/DAM-3B-Self-Contained',
            trust_remote_code=True,
            torch_dtype=torch.float16).to(self.device)
        self.dam = self.dam_model.init_dam(conv_mode='v1', prompt_mode='full+focal_crop')
    
    def apply_sam(self, image, input_points=None, input_boxes=None, input_labels=None):
        """Apply SAM to generate masks"""
        inputs = self.sam_processor(
            image, 
            input_points=input_points, 
            input_boxes=input_boxes,
            input_labels=input_labels, 
            return_tensors="pt" ).to(self.device)

        with torch.no_grad():
            outputs = self.sam_model(**inputs)

        masks = self.sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu())[0][0]
        scores = outputs.iou_scores[0, 0]

        mask_selection_index = scores.argmax()
        return masks[mask_selection_index].numpy()
    
    # Rimosso generate_description, tanto adesso ho generate_description_with_presets
    
    def add_contour(self, img, mask, input_points=None, input_boxes=None):
        """Add visualization contours to image"""
        img = img.copy()
        mask = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (1.0, 1.0, 1.0), thickness=6)

        if input_points is not None:
            for points in input_points:
                for x, y in points:
                    cv2.circle(img, (int(x), int(y)), radius=10, color=(1.0, 0.0, 0.0), thickness=-1)
                    cv2.circle(img, (int(x), int(y)), radius=10, color=(1.0, 1.0, 1.0), thickness=2)

        if input_boxes is not None:
            for box_batch in input_boxes:
                for box in box_batch:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color=(1.0, 1.0, 1.0), thickness=4)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color=(1.0, 0.0, 0.0), thickness=2)

        return img

    # rimosso generate_multiple_descriptions()
        
    def generate_description_with_preset(self, image, mask, preset='balanced'):
        """Generate single description using a specific preset"""
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        params = self.PRESET_PARAMS[preset]
        
        description = []
        for token in self.dam.get_description(
            image,
            mask_img,
            '<image>\nDescribe the masked region in detail.', # prompt base
            streaming=True,
            **params
        ):
            description.append(token)
        
        return ''.join(description)

def generate_grid_points(width: int, height: int, level: int) -> List[List[int]]:
    """Generate grid points based on level of detail"""
    if level < 0:
        raise ValueError("Level must be >= 0")
        
    if level == 0: # livello 0 prende 1x1 cioè un punto al centro dell'immagine
        return [[width // 2, height // 2]]
    
    points = []
    grid_size = level + 2
    
    for i in range(1, grid_size):
        for j in range(1, grid_size):
            x = width * i // grid_size
            y = height * j // grid_size
            points.append([x, y])
    
    return points

def process_single_image(
    image_path: str, 
    output_path: str, 
    dam_processor: DAMProcessor, 
    level: int
    ) -> dict:
    """Process single image and return descriptions for ALL presets"""
    with Image.open(image_path) as img:
        width, height = img.size
        input_points = generate_grid_points(width, height, level)
        mask = dam_processor.apply_sam(img, input_points=[input_points], input_labels=[[1]*len(input_points)])
        
        # Generate descriptions for all presets
        descriptions = {
            'conservative': dam_processor.generate_description_with_preset(img, mask, 'conservative'),
            'balanced': dam_processor.generate_description_with_preset(img, mask, 'balanced'),
            'creative': dam_processor.generate_description_with_preset(img, mask, 'creative')
        }
        
        # Save visualization
        img_np = np.asarray(img).astype(float) / 255.0
        img_with_contour = dam_processor.add_contour(img_np, mask, input_points=[input_points])
        Image.fromarray((img_with_contour * 255).astype(np.uint8)).save(output_path)
        
        return descriptions

def process_image_folder(
    image_folder: str,
    output_root: str,
    dam_processor: DAMProcessor,
    max_level: int = 4
    ) -> dict:
    """Process all images with ALL presets for each grid level"""
    all_descriptions = {}  # {filename: {level: {preset: description}}}
    
    for level in range(max_level + 1):
        mode_name = f"grid_level_{level}"
        print(f"\n== Processing Grid Level: {level} ==")
        
        output_folder = os.path.join(output_root, mode_name)
        os.makedirs(output_folder, exist_ok=True)

        for filename in os.listdir(image_folder):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_path = os.path.join(image_folder, filename)
                base_name = os.path.splitext(filename)[0]
                
                if base_name not in all_descriptions:
                    all_descriptions[base_name] = {}
                
                output_image_path = os.path.join(output_folder, f"out_{filename}")
                text_output_path = os.path.join(output_folder, f"descriptions_{filename}.txt")

                print(f"Processing {filename} with {((level + 1) ** 2) if level > 0 else 1} points...")

                # Get descriptions for all presets
                descriptions = process_single_image(
                    image_path, output_image_path, dam_processor, level
                )
                
                # Store in the main dictionary
                all_descriptions[base_name][level] = descriptions
                
                # Save to text file
                with open(text_output_path, "w", encoding="utf-8") as f:
                    for preset, desc in descriptions.items():
                        f.write(f"=== {preset.capitalize()} ===\n{desc}\n\n")
    
    return all_descriptions


def main():
    config = {
        "image_folder": "test_fatti_autoDAM/images_test",
        "output_root": "test_fatti_autoDAM/output_test",
        "max_level": 4
    }

    print("Initializing models...")
    dam_processor = DAMProcessor()

    print("\nStarting DAM grid point sampling experiment...")
    results = process_image_folder(
        image_folder=config["image_folder"],
        output_root=config["output_root"],
        dam_processor=dam_processor,
        max_level=config["max_level"]
    )
    
    # Example of accessing results:
    for filename, level_data in results.items():
        print(f"\nDescriptions for {filename}:")
        for level, presets in level_data.items():
            print(f"  Level {level}:")
            for preset, desc in presets.items():
                print(f"    {preset}: {desc}")  # Print the descriptions

    print("\nExperiment completed!")

if __name__ == "__main__":
    main()