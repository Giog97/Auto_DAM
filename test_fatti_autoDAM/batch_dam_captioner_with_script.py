"""
Wrapper Python per lanciare dam_with_sam_self_contained.py, in modo da:
- Iterare su tutte le immagini in una cartella.
- Costruire e chiamare il comando per dam_with_sam_self_contained.py via subprocess.
"""

import os
import time
import random
from PIL import Image
import subprocess
from typing import List, Tuple

def generate_grid_points(width: int, height: int, level: int) -> List[List[int]]:
    """
    Generate grid points based on the level of detail.
    
    Args:
        width: Image width in pixels
        height: Image height in pixels
        level: Grid level (0=center point, 1=2x2 grid, 2=3x3 grid, etc.)
    
    Returns:
        List of [x, y] coordinate pairs
    """
    if level < 0:
        raise ValueError("Level must be >= 0")
        
    # Special case for level 0 (just center point)
    if level == 0:
        return [[width // 2, height // 2]]
    
    points = []
    grid_size = level + 2  # Level 1 → 3x3 grid (9 points), etc.
    
    # Create grid points
    for i in range(1, grid_size):
        for j in range(1, grid_size):
            x = width * i // grid_size
            y = height * j // grid_size
            points.append([x, y])
    
    return points

def process_single_image(image_path: str, output_path: str, script_path: str, level: int) -> str:
    """
    Process a single image with DAM using grid points at specified level.
    
    Args:
        image_path: Path to input image
        output_path: Path to save output image
        script_path: Path to DAM script
        level: Grid level to use
    
    Returns:
        Extracted description from DAM
    """
    # Get image dimensions
    with Image.open(image_path) as img:
        width, height = img.size

    # Generate grid points
    input_data = generate_grid_points(width, height, level)

    # Build and execute DAM command
    command = [
        "python", script_path,
        "--image_path", image_path,
        "--output_image_path", output_path,
        "--points", str(input_data)
    ]

    result = subprocess.run(command, capture_output=True, text=True)
    return extract_description(result.stdout)

def extract_description(dam_output: str) -> str:
    """
    Extract description text from DAM output.
    
    Args:
        dam_output: String containing DAM's stdout
    
    Returns:
        Extracted description text
    """
    lines = dam_output.splitlines()
    description = "[Description not found]"
    
    for i, line in enumerate(lines):
        if line.strip().startswith("Description:"):
            description_lines = []
            for line in lines[i + 1:]:
                if line.strip().startswith("Output image"):
                    break
                description_lines.append(line)
            description = "\n".join(description_lines).strip()
            break
            
    return description

def process_image_folder(
    image_folder: str,
    output_root: str,
    script_path: str,
    max_level: int = 5,
    image_extensions: Tuple[str] = ('.jpg', '.jpeg', '.png', '.bmp')
) -> None:
    """
    Process all images in folder with multiple grid levels.
    
    Args:
        image_folder: Path to folder containing images
        output_root: Base path for output folders
        script_path: Path to DAM script
        max_level: Maximum grid level to test (inclusive)
        image_extensions: Tuple of valid image extensions
    """
    for level in range(max_level + 1):
        mode_name = f"grid_level_{level}"
        print(f"\n== Processing Grid Level: {level} ==")
        
        # Create output folder for this level
        output_folder = os.path.join(output_root, mode_name)
        os.makedirs(output_folder, exist_ok=True)

        # Start timing
        start_time = time.time()
        image_count = 0

        # Process each image
        for filename in os.listdir(image_folder):
            if filename.lower().endswith(image_extensions):
                image_count += 1
                image_path = os.path.join(image_folder, filename)
                output_image_path = os.path.join(output_folder, f"out_{filename}")
                text_output_path = os.path.join(output_folder, f"description_{filename}.txt")

                print(f"Processing {filename} with {((level + 1) ** 2) if level > 0 else 1} points...")

                # Process image and get description
                description = process_single_image(
                    image_path, output_image_path, script_path, level
                )

                # Save description
                with open(text_output_path, "w", encoding="utf-8") as f:
                    f.write(description)

        # Save timing info
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
    """Main function to configure and run the processing."""
    # Configuration
    config = {
        "image_folder": "test_fatti_autoDAM/images_test",
        "output_root": "test_fatti_autoDAM/output_test",
        "script_path": "examples/dam_with_sam_self_contained.py",
        "max_level": 5  # Maximum grid level to test
    }

    # Validate paths
    for path in [config["image_folder"], config["script_path"]]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path does not exist: {path}")

    # Run processing
    print("Starting DAM grid point sampling experiment...")
    process_image_folder(
        image_folder=config["image_folder"],
        output_root=config["output_root"],
        script_path=config["script_path"],
        max_level=config["max_level"]
    )
    print("\nExperiment completed!")

if __name__ == "__main__":
    main()

"""
# CONFIGURAZIONE BASE
image_folder = "images_test"
output_root = "output_test"
script_path = "examples/dam_with_sam_self_contained.py"
"""
