import os
import time
import random
import numpy as np
import torch
from PIL import Image
from transformers import SamModel, SamProcessor, AutoModel
import cv2
from typing import List, Tuple

class DAMProcessor:
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
            torch_dtype=torch.float16
        ).to(self.device)
        self.dam = self.dam_model.init_dam(conv_mode='v1', prompt_mode='full+focal_crop')
    
    def apply_sam(self, image, input_points=None, input_boxes=None, input_labels=None):
        """Apply SAM to generate masks"""
        inputs = self.sam_processor(
            image, 
            input_points=input_points, 
            input_boxes=input_boxes,
            input_labels=input_labels, 
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.sam_model(**inputs)

        masks = self.sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu()
        )[0][0]
        scores = outputs.iou_scores[0, 0]

        mask_selection_index = scores.argmax()
        return masks[mask_selection_index].numpy()
    
    def generate_description(self, image, mask, prompt='<image>\nDescribe the masked region in detail.'):
        """Generate description using DAM"""
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        description = []
        
        for token in self.dam.get_description(
            image,
            mask_img,
            prompt,
            streaming=True,
            temperature=0.2, # controls randomness (you can increase it to get more variety)
            top_p=0.5, # this enables nucleus sampling
            num_beams=1, # disables beam search (important!)
            max_new_tokens=512
        ):
            description.append(token)
            
        return ''.join(description)
    
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

def process_single_image(image_path: str, output_path: str, dam_processor: DAMProcessor, level: int) -> str:
    """Process single image with DAM at specified grid level"""
    with Image.open(image_path) as img:
        width, height = img.size
        input_points = generate_grid_points(width, height, level)
        
        # Get mask from SAM
        mask = dam_processor.apply_sam(img, input_points=[input_points], input_labels=[[1]*len(input_points)])
        
        # Generate description
        description = dam_processor.generate_description(img, mask)
        
        # Save visualization
        img_np = np.asarray(img).astype(float) / 255.0
        img_with_contour = dam_processor.add_contour(img_np, mask, input_points=[input_points])
        Image.fromarray((img_with_contour * 255).astype(np.uint8)).save(output_path)
        
        return description

def process_image_folder(
    image_folder: str,
    output_root: str,
    dam_processor: DAMProcessor,
    max_level: int = 5,
    image_extensions: Tuple[str] = ('.jpg', '.jpeg', '.png', '.bmp')
) -> None:
    """Process all images in folder with multiple grid levels"""
    for level in range(max_level + 1):
        mode_name = f"grid_level_{level}"
        print(f"\n== Processing Grid Level: {level} ==")
        
        output_folder = os.path.join(output_root, mode_name)
        os.makedirs(output_folder, exist_ok=True)

        start_time = time.time()
        image_count = 0

        for filename in os.listdir(image_folder):
            if filename.lower().endswith(image_extensions):
                image_count += 1
                image_path = os.path.join(image_folder, filename)
                output_image_path = os.path.join(output_folder, f"out_{filename}")
                text_output_path = os.path.join(output_folder, f"description_{filename}.txt")

                print(f"Processing {filename} with {((level + 1) ** 2) if level > 0 else 1} points...")

                description = process_single_image(
                    image_path, output_image_path, dam_processor, level
                )

                with open(text_output_path, "w", encoding="utf-8") as f:
                    f.write(description)

        total_time = time.time() - start_time
        avg_time = total_time / image_count if image_count > 0 else 0
        
        with open(os.path.join(output_folder, "timing.txt"), "w", encoding="utf-8") as f:
            f.write(f"Grid Level: {level}\n")
            f.write(f"Number of points: {(level + 1) ** 2 if level > 0 else 1}\n")
            f.write(f"Number of images: {image_count}\n")
            f.write(f"Total time: {total_time:.2f} seconds\n")
            f.write(f"Average time per image: {avg_time:.2f} seconds\n")

        print(f"\nCompleted level {level}:")
        print(f"- Processed {image_count} images")
        print(f"- Total time: {total_time:.2f} seconds")
        print(f"- Average time per image: {avg_time:.2f} seconds")

def main():
    """Main function to configure and run the processing"""
    config = {
        "image_folder": "test_fatti_autoDAM/images_test",
        "output_root": "test_fatti_autoDAM/output_test",
        "max_level": 5
    }

    for path in [config["image_folder"]]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path does not exist: {path}")

    print("Initializing models...")
    dam_processor = DAMProcessor()

    print("\nStarting DAM grid point sampling experiment...")
    process_image_folder(
        image_folder=config["image_folder"],
        output_root=config["output_root"],
        dam_processor=dam_processor,
        max_level=config["max_level"]
    )
    print("\nExperiment completed!")

if __name__ == "__main__":
    main()