# Copyright 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

# Running this script does not require installing `dam` package

import argparse
import ast
import torch
import numpy as np
from PIL import Image
from transformers import SamModel, SamProcessor, AutoModel
import cv2
import os

# Serve se passiamo punti o box
def apply_sam(image, input_points=None, input_boxes=None, input_labels=None):
    inputs = sam_processor(image, input_points=input_points, input_boxes=input_boxes,
                           input_labels=input_labels, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = sam_model(**inputs)

    masks = sam_processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu()
    )[0][0]
    scores = outputs.iou_scores[0, 0]

    mask_selection_index = scores.argmax()

    mask_np = masks[mask_selection_index].numpy()

    return mask_np

def add_contour(img, mask, input_points=None, input_boxes=None):
    img = img.copy()

    # Draw contour
    mask = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (1.0, 1.0, 1.0), thickness=6)

    # Draw points if provided
    if input_points is not None:
        for points in input_points:  # Handle batch of points
            for x, y in points:
                cv2.circle(img, (int(x), int(y)), radius=10,
                           color=(1.0, 0.0, 0.0), thickness=-1)
                cv2.circle(img, (int(x), int(y)), radius=10,
                           color=(1.0, 1.0, 1.0), thickness=2)

    # Draw boxes if provided
    if input_boxes is not None:
        for box_batch in input_boxes:
            for box in box_batch:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img, (x1, y1), (x2, y2),
                              color=(1.0, 1.0, 1.0), thickness=4)
                cv2.rectangle(img, (x1, y1), (x2, y2),
                              color=(1.0, 0.0, 0.0), thickness=2)

    return img

def denormalize_coordinates(coords, image_size, is_box=False):
    width, height = image_size
    if is_box:
        x1, y1, x2, y2 = coords
        return [
            int(x1 * width),
            int(y1 * height),
            int(x2 * width),
            int(y2 * height)
        ]
    else:
        x, y = coords
        return [int(x * width), int(y * height)]

def print_streaming(text):
    print(text, end="", flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Describe Anything script")
    parser.add_argument('--image_path', type=str,
                        required=True, help='Path to the image file')
    parser.add_argument('--mask_path', type=str,
                        help='Path to precomputed binary mask image (PNG) or .npy file')
    parser.add_argument('--points', type=str,
                        default='[[1172, 812], [1572, 800]]', help='List of points for SAM input')
    parser.add_argument('--box', type=str,
                        default='[773, 518, 1172, 812]', help='Bounding box for SAM input')
    parser.add_argument('--use_box', action='store_true',
                        help='Use box instead of points for SAM input')
    parser.add_argument('--query', type=str,
                        default='<image>\nDescribe the masked region in detail.', help='Prompt for the model')
    parser.add_argument('--model_path', type=str,
                        default='nvidia/DAM-3B-Self-Contained', help='Path to the model checkpoint')
    parser.add_argument('--prompt_mode', type=str,
                        default='focal_prompt', help='Prompt mode')
    parser.add_argument('--conv_mode', type=str,
                        default='v1', help='Conversation mode')
    parser.add_argument('--temperature', type=float,
                        default=0.2, help='Sampling temperature')
    parser.add_argument('--top_p', type=float,
                        default=0.5, help='Top-p for sampling')
    parser.add_argument('--output_image_path', type=str, default=None,
                        help='Path to save the output image with contour')
    parser.add_argument('--normalized_coords', action='store_true',
                        help='Interpret coordinates as normalized (0-1) values')
    parser.add_argument('--no_stream', action='store_true',
                        help='Disable streaming output')

    args = parser.parse_args()

    # Load the image
    img = Image.open(args.image_path).convert('RGB')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam_model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
    sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

    image_size = img.size  # (width, height)

    # Load or compute the mask
    if args.mask_path:
        # Se la maschera è presente la carica (controlla se è un file .npy o no)
        print(f"Loading mask from {args.mask_path}")
        if args.mask_path.endswith('.npy'):
            mask_np = np.load(args.mask_path)
        else:
            mask_image = Image.open(args.mask_path).convert('L')  # grayscale
            mask_np = np.array(mask_image)
            mask_np = (mask_np > 0).astype(np.uint8)
    else:
        # Se la maschera non è presente la calcola usando i punti o i box
        if args.use_box:
            input_boxes = ast.literal_eval(args.box)
            if args.normalized_coords:
                input_boxes = denormalize_coordinates(
                    input_boxes, image_size, is_box=True)
            input_boxes = [[input_boxes]]
            print(f"Using input_boxes: {input_boxes}")
            mask_np = apply_sam(img, input_boxes=input_boxes)
        else:
            input_points = ast.literal_eval(args.points)
            if args.normalized_coords:
                input_points = [denormalize_coordinates(point, image_size)
                                for point in input_points]
            input_labels = [1] * len(input_points)
            input_points = [[x, y] for x, y in input_points]
            input_points = [input_points]
            input_labels = [input_labels]
            print(f"Using input_points: {input_points}")
            mask_np = apply_sam(img, input_points=input_points,
                                input_labels=input_labels)

    # Ensure mask is binary
    mask = Image.fromarray((mask_np * 255).astype(np.uint8)) 

    # Initialize DAM model
    prompt_modes = {
        "focal_prompt": "full+focal_crop",
    }

    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype='torch.float16').to(device)
    dam = model.init_dam(conv_mode=args.conv_mode, prompt_mode=prompt_modes.get(args.prompt_mode, args.prompt_mode))

    # Get description
    print("Description:")
    if not args.no_stream:
        for token in dam.get_description(
                img,
                mask,
                args.query,
                streaming=True,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=1,
                max_new_tokens=512
        ):
            print_streaming(token)
        print()
    else:
        outputs = dam.get_description(
            img,
            mask,
            args.query,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=1,
            max_new_tokens=512,
        )
        print(f"Description:\n{outputs}")

    if args.output_image_path:
        img_np = np.asarray(img).astype(float) / 255.0

        vis_points = input_points if not args.use_box and not args.mask_path else None
        vis_boxes = input_boxes if args.use_box and not args.mask_path else None

        img_with_contour_np = add_contour(img_np, mask_np,
                                          input_points=vis_points,
                                          input_boxes=vis_boxes)
        img_with_contour_pil = Image.fromarray(
            (img_with_contour_np * 255.0).astype(np.uint8))
        img_with_contour_pil.save(args.output_image_path)
        print(f"Output image with contour saved as {args.output_image_path}")

