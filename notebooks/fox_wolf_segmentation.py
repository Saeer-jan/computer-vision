
# ============================================================================
# IMPORTS
# ============================================================================
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime


# ============================================================================
# MODEL INITIALIZATION
# ============================================================================
def load_model(model_path):
    """Load the trained YOLO model"""
    try:
        model = YOLO(model_path)
        print(f"✓ Model loaded successfully!")
        print(f"  Model path: {model_path}")
        print(f"  Model task: {model.task}")
        print(f"  Model classes: {model.names}")
        return model
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None


# ============================================================================
# SINGLE IMAGE PREDICTION
# ============================================================================
def predict_and_visualize(model, image_path, conf_threshold=0.5):
    """
    Run YOLO inference on an image and visualize results
    
    Args:
        model: YOLO model object
        image_path: Path to the image file
        conf_threshold: Confidence threshold for detections (0-1)
    """
    try:
        # Check if file exists
        if not Path(image_path).exists():
            print(f"✗ Image not found at: {image_path}")
            return None
        
        # Run inference
        results = model.predict(image_path, conf=conf_threshold, verbose=False)
        result = results[0]
        
        # Load and display image with annotations
        img = cv2.imread(image_path)
        if img is None:
            print(f"✗ Could not read image: {image_path}")
            return None
        
        # Get annotated image
        annotated_img = result.plot()
        annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        
        # Display
        plt.figure(figsize=(12, 8))
        plt.imshow(annotated_img_rgb)
        plt.axis('off')
        plt.title(f"Predictions - {Path(image_path).name} (Confidence > {conf_threshold})")
        plt.tight_layout()
        plt.show()
        
        # Print detection results
        print(f"\n{'='*60}")
        print(f"IMAGE: {Path(image_path).name}")
        print(f"Detections found: {len(result.boxes)}")
        print(f"{'='*60}")
        
        if len(result.boxes) == 0:
            print("No objects detected")
        else:
            for idx, box in enumerate(result.boxes):
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])
                print(f"  {idx+1}. {class_name.upper()} - Confidence: {confidence:.2%}")
        
        return result
    
    except Exception as e:
        print(f"✗ Error processing image: {e}")
        return None


# ============================================================================
# BATCH PROCESSING
# ============================================================================
def batch_predict(model, folder_path, conf_threshold=0.5):
    """
    Run inference on all images in a folder
    
    Args:
        model: YOLO model object
        folder_path: Path to folder containing images
        conf_threshold: Confidence threshold for detections (0-1)
    
    Returns:
        List of results dictionaries
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.avif', '.webp'}
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"✗ Folder not found: {folder_path}")
        return []
    
    images = sorted([f for f in folder.glob('*') if f.suffix.lower() in image_extensions])
    
    if not images:
        print(f"⚠️  No images found in {folder_path}")
        return []
    
    print(f"\n{'='*60}")
    print(f"BATCH PROCESSING")
    print(f"{'='*60}")
    print(f"Found {len(images)} images. Processing...")
    
    all_results = []
    for idx, img_path in enumerate(images, 1):
        print(f"  [{idx}/{len(images)}] Processing: {img_path.name}...", end=" ")
        try:
            results = model.predict(str(img_path), conf=conf_threshold, verbose=False)
            detection_count = len(results[0].boxes)
            classes = [model.names[int(box.cls[0])] for box in results[0].boxes]
            all_results.append({
                'image': str(img_path),
                'detections': detection_count,
                'classes': classes
            })
            print(f"✓ ({detection_count} detections)")
        except Exception as e:
            print(f"✗ Error")
            all_results.append({
                'image': str(img_path),
                'detections': 0,
                'classes': [],
                'error': str(e)
            })
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"BATCH PROCESSING RESULTS SUMMARY")
    print(f"{'='*60}")
    
    total_detections = 0
    for result in all_results:
        print(f"\n📸 {Path(result['image']).name}")
        print(f"   Detections: {result['detections']}")
        if result['detections'] > 0:
            class_summary = {}
            for cls in result['classes']:
                class_summary[cls] = class_summary.get(cls, 0) + 1
            print(f"   Classes: {', '.join([f'{c} ({count})' for c, count in class_summary.items()])}")
        total_detections += result['detections']
    
    print(f"\n{'='*60}")
    print(f"Total images processed: {len(all_results)}")
    print(f"Total detections: {total_detections}")
    print(f"{'='*60}\n")
    
    return all_results


# ============================================================================
# EXPORT RESULTS
# ============================================================================
def export_predictions(model, image_path, output_folder="./predictions"):
    """
    Export predictions with annotated image and segmentation masks
    
    Args:
        model: YOLO model object
        image_path: Path to input image
        output_folder: Folder to save results
    
    Returns:
        Path to output folder
    """
    try:
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Run inference
        results = model.predict(image_path, verbose=False)
        result = results[0]
        
        # Save annotated image
        result.save(str(output_path / f"{Path(image_path).stem}_annotated.jpg"))
        print(f"✓ Saved: {output_path / f'{Path(image_path).stem}_annotated.jpg'}")
        
        # Save segmentation masks if available
        if result.masks is not None:
            for idx, mask in enumerate(result.masks):
                mask_array = mask.data.cpu().numpy()[0] * 255
                mask_path = output_path / f"{Path(image_path).stem}_mask_{idx}.png"
                cv2.imwrite(str(mask_path), mask_array.astype(np.uint8))
                print(f"✓ Saved: {mask_path}")
        
        print(f"✓ Results saved to: {output_path}\n")
        return output_path
    
    except Exception as e:
        print(f"✗ Error exporting predictions: {e}")
        return None


# ============================================================================
# DISPLAY MODEL INFO
# ============================================================================
def display_model_info(model):
    """Display model information"""
    print(f"\n{'='*60}")
    print(f"MODEL INFORMATION")
    print(f"{'='*60}")
    print(f"Task: {model.task}")
    print(f"Classes: {model.names}")
    print(f"Number of classes: {len(model.names)}")
    print(f"{'='*60}\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """Main execution function"""
    
    # Model configuration
    MODEL_PATH = r"C:\Users\Saeer Jan\Downloads\best_segmentation_model (2).pt"
    
    # Load model
    model = load_model(MODEL_PATH)
    if model is None:
        return
    
    # Display model info
    display_model_info(model)
    
    # ========================================================================
    # OPTION 1: Single Image Prediction
    # ========================================================================
    print("\n" + "="*60)
    print("TESTING ON SINGLE IMAGES")
    print("="*60)
    
    # Test on individual images
    test_images = [
        r"C:\Users\Saeer Jan\Downloads\red-fox-portrait.avif",
        r"C:\Users\Saeer Jan\Downloads\image1.jpg",
        r"C:\Users\Saeer Jan\Downloads\Gray-wolf.webp",
        r"C:\Users\Saeer Jan\Downloads\wol and fo.webp"
    ]
    
    for image_path in test_images:
        print(f"\nProcessing: {image_path}")
        if Path(image_path).exists():
            predict_and_visualize(model, image_path, conf_threshold=0.5)
            # Optional: Export results
            # export_predictions(model, image_path, output_folder="./predictions")
        else:
            print(f"✗ Image not found: {image_path}")
    
    # ========================================================================
    # OPTION 2: Batch Processing
    # ========================================================================
    print("\n" + "="*60)
    print("BATCH PROCESSING FOLDER")
    print("="*60)
    
    folder_path = r"C:\Users\Saeer Jan\Downloads\New folder"
    if Path(folder_path).exists():
        batch_results = batch_predict(model, folder_path, conf_threshold=0.5)
    else:
        print(f"✗ Folder not found: {folder_path}")
    
    print("\n✓ Script execution completed!")


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    main()
