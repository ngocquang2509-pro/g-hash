"""
Educational CBIR Pipeline V2: Extraction, YOLO Cropping, Deduplication & GPU CLIP
---------------------------------------------------------------------------------------------------------
Dependencies:
    pip install opencv-python ultralytics transformers torch pandas tqdm Pillow imagehash

Usage:
    python prepare_video_dataset.py --video_dir data/ET-EDU --output_dir data/cropped_students --fps 1.0 --hash_threshold 7 --threshold 0.3 --debug
"""

import os
import cv2
import csv
import torch
import argparse
import imagehash
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Import ML Libraries
import warnings
warnings.filterwarnings("ignore")
from ultralytics import YOLO
from transformers import pipeline

BEHAVIORAL_CLASSES = [
    'using phone', 'dozing off', 'turning sideways', 
    'turning vertically', 'fighting', 'hugging', 
    'raising hand', 'opening book', 'reading', 'taking notes'
]

def parse_args():
    parser = argparse.ArgumentParser(description="Extract frames, crop students, deduplicate via Hash, and annotate with CLIP.")
    parser.add_argument("--video_dir", type=str, default="data/ET-EDU", help="Directory containing raw classroom videos")
    parser.add_argument("--output_dir", type=str, default="data/cropped_students", help="Directory to save cropped person images")
    parser.add_argument("--csv_file", type=str, default="data/draft_annotations.csv", help="Path to save the generated multi-label CSV")
    parser.add_argument("--fps", type=float, default=1.0, help="Frames per second to extract (e.g., 0.5 for 1 frame every 2 seconds)")
    parser.add_argument("--min_size", type=int, default=50, help="Minimum width and height in pixels for valid bounding boxes")
    parser.add_argument("--hash_threshold", type=int, default=7, help="Max Hamming distance to be considered a duplicate. Recommended: 5-8")
    parser.add_argument("--threshold", type=float, default=0.25, help="Confidence threshold for CLIP multi-label assignment")
    parser.add_argument("--debug", action="store_true", help="Enable cv2.imshow preview of cropped bounding boxes")
    return parser.parse_args()

def init_csv(csv_path):
    """Initializes the CSV with proper headers"""
    headers = ['Image_Path'] + [cls.replace(' ', '_') for cls in BEHAVIORAL_CLASSES]
    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)

def append_to_csv(csv_path, image_path, binary_predictions):
    """Appends a single row to the CSV file"""
    row = [image_path] + binary_predictions
    with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(row)

def main():
    args = parse_args()
    
    # 1. Setup Directories
    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(args.csv_file).parent.mkdir(parents=True, exist_ok=True)
    
    if not video_dir.exists():
        print(f"❌ Error: Video directory '{video_dir}' not found.")
        return
    
    # 2. Load AI Models
    print("[*] Loading YOLOv8n for Person Detection...")
    yolo_model = YOLO("yolov8n.pt")
    
    print("[*] Loading CLIP (openai/clip-vit-base-patch32) for Zero-Shot Multi-Labeling...")
    device = 0 if torch.cuda.is_available() else -1
    clip_classifier = pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32", device=device)
    
    init_csv(args.csv_file)
    print(f"[*] CSV initialized at '{args.csv_file}'")
    
    # 3. Find Videos
    valid_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    video_files = [f for f in video_dir.iterdir() if f.suffix.lower() in valid_extensions]
    print(f"[*] Found {len(video_files)} videos in {video_dir}")
    
    # Process each video
    for video_idx, video_path in enumerate(video_files, 1):
        print(f"\n🎥 Processing Video {video_idx}/{len(video_files)}: {video_path.name}")
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"❌ Warning: Could not open {video_path.name}")
            continue
            
        v_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if v_fps <= 0: v_fps = 30
        frame_interval = max(1, int(v_fps / args.fps))
        
        # In-memory Hash Log for current video deduplication
        seen_hashes = []
        global_duplicate_count = 0
        
        frame_id = 0
        pbar = tqdm(total=total_frames, desc="Extracting frames")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_id % frame_interval == 0:
                # A: Detect Persons (class 0)
                results = yolo_model.predict(frame, classes=[0], verbose=False)
                
                if len(results) > 0 and len(results[0].boxes) > 0:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    
                    for person_idx, box in enumerate(boxes, 1):
                        x1, y1, x2, y2 = map(int, box[:4])
                        w, h = (x2 - x1), (y2 - y1)
                        
                        # B: Filter Noise
                        if w < args.min_size or h < args.min_size:
                            continue
                            
                        # Crop bounding box
                        person_crop = frame[y1:y2, x1:x2]
                        
                        # Convert to PIL for Hash and CLIP
                        pil_img = Image.fromarray(cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB))
                        
                        # C: DEDUPLICATION using Perceptual Hash (phash)
                        crop_hash = imagehash.phash(pil_img)
                        is_duplicate = False
                        
                        # Scan backwards through seen memory
                        for h_past in seen_hashes:
                            if crop_hash - h_past <= args.hash_threshold:
                                is_duplicate = True
                                break
                                
                        if is_duplicate:
                            global_duplicate_count += 1
                            continue # Skip Saving and CLIP inference to save massive compute
                            
                        # Log hash to block future exact copies
                        seen_hashes.append(crop_hash)
                        
                        # D: Filename formulation & Saving
                        img_filename = f"{video_path.stem}_frame{frame_id:06d}_person{person_idx:03d}.jpg"
                        img_save_path = output_dir / img_filename
                        cv2.imwrite(str(img_save_path), person_crop)
                        
                        # Debug GUI mode
                        if args.debug:
                            cv2.imshow("Debug: Cropped Person", person_crop)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                args.debug = False
                                cv2.destroyAllWindows()
                                
                        # E: CLIP Zero-Shot Classification
                        prompts = [f"a student {beh}" for beh in BEHAVIORAL_CLASSES]
                        try:
                            clip_results = clip_classifier(pil_img, candidate_labels=prompts, multi_label=True)
                            score_map = {res['label'].replace("a student ", ""): res['score'] for res in clip_results}
                            
                            binary_predictions = []
                            for beh in BEHAVIORAL_CLASSES:
                                score = score_map.get(beh, 0.0)
                                binary_predictions.append(1 if score >= args.threshold else 0)
                                
                            # F: Log to CSV
                            rel_path = str(Path(args.output_dir).name) + "/" + img_filename
                            append_to_csv(args.csv_file, rel_path, binary_predictions)
                                
                        except Exception as e:
                            print(f"\n⚠️ CLIP inference failed for {img_filename}: {e}")
            
            frame_id += 1
            pbar.update(1)
            
        pbar.close()
        cap.release()
        print(f"   -> Blocked {global_duplicate_count} heavily identical duplicate crops using phash.")
        
    if args.debug:
        cv2.destroyAllWindows()
        
    print(f"\n🎉 Pipeline Completed! Draft annotations saved to: {args.csv_file}")

if __name__ == "__main__":
    main()
