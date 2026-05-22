import os
import json
import torch
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.transforms import functional as F
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from pycocotools.coco import COCO
import cv2
import warnings

warnings.filterwarnings('ignore')

# ========= CONFIGURATION =========
MODEL_PATH = "final_maskrcnn.pth"  # Your trained model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = (256, 256)
CONFIDENCE_THRESHOLD = 0.7  # Based on your evaluation results

# Dataset paths
CHALLENGE_CONFIG = {
    "image_dir": "D:/Deepfake/datasets/Test-Challenge",
    "ann_file": "D:/Deepfake/datasets/Annotations/Test-Challenge_poly_cleaned.json"
}

OUTPUT_DIR = "demo_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Class mapping
CLASS_NAMES = {0: 'background', 1: 'fake', 2: 'real'}
CLASS_COLORS = {1: (255, 0, 0), 2: (0, 255, 0)}  # Red for fake, Green for real

# Enable interactive mode for PyCharm
plt.ion()  # Turn on interactive mode


# ========= MODEL SETUP =========
def get_model(num_classes=3):
    """Create model with same architecture as training"""
    model = maskrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    return model


def load_model(model_path, device):
    """Load the trained model"""
    print(f"🔧 Loading model from {model_path}...")
    model = get_model(num_classes=3)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    model.to(device)
    print("✅ Model loaded successfully!")
    return model


# ========= IMAGE PROCESSING =========
def preprocess_image(image_path):
    """Preprocess image for model input"""
    img = Image.open(image_path).convert("RGB")
    original_size = img.size
    img_resized = img.resize(IMAGE_SIZE)
    tensor = F.to_tensor(img_resized).unsqueeze(0)
    return img, img_resized, tensor, original_size


def predict_image(model, tensor, device, threshold=CONFIDENCE_THRESHOLD):
    """Get model predictions for an image"""
    tensor = tensor.to(device)

    with torch.no_grad():
        outputs = model(tensor)[0]

    # Filter predictions by confidence threshold
    valid_indices = outputs['scores'] > threshold

    predictions = {
        'boxes': outputs['boxes'][valid_indices].cpu().numpy(),
        'labels': outputs['labels'][valid_indices].cpu().numpy(),
        'scores': outputs['scores'][valid_indices].cpu().numpy(),
        'masks': outputs['masks'][valid_indices].cpu().numpy()
    }

    return predictions


# ========= ENHANCED VISUALIZATION =========
def display_interactive_results(original_img, predictions, ground_truth_info=None, image_name="Unknown"):
    """Create and display interactive visualization with all details"""

    # Create a larger figure for better visibility
    fig = plt.figure(figsize=(20, 15))
    fig.suptitle(f'🔍 Deepfake Detection Results - {image_name}', fontsize=20, fontweight='bold')

    # Convert PIL to numpy for visualization
    img_np = np.array(original_img.resize(IMAGE_SIZE))

    # Create a 2x3 grid for more detailed visualization
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Original Image (Large)
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.imshow(img_np)
    ax1.set_title('📸 Original Image', fontsize=16, fontweight='bold')
    ax1.axis('off')

    # 2. Bounding Boxes
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.imshow(img_np)

    detection_info = []
    for i, (box, label, score) in enumerate(zip(predictions['boxes'], predictions['labels'], predictions['scores'])):
        x1, y1, x2, y2 = box
        width, height = x2 - x1, y2 - y1

        color = 'red' if label == 1 else 'green'
        rect = patches.Rectangle((x1, y1), width, height, linewidth=3,
                                 edgecolor=color, facecolor='none')
        ax2.add_patch(rect)

        # Add label with background
        class_name = CLASS_NAMES[label]
        ax2.text(x1, y1 - 5, f'{class_name}: {score:.3f}',
                 color='white', fontweight='bold', fontsize=12,
                 bbox=dict(boxstyle="round,pad=0.5", facecolor=color, alpha=0.8))

        detection_info.append(f"{class_name.upper()}: {score:.3f}")

    ax2.set_title('🎯 Bounding Boxes', fontsize=14, fontweight='bold')
    ax2.axis('off')

    # 3. Segmentation Masks
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.imshow(img_np)

    if len(predictions['masks']) > 0:
        # Create overlay for masks
        mask_overlay = np.zeros_like(img_np, dtype=np.float32)

        for mask, label in zip(predictions['masks'], predictions['labels']):
            mask_binary = mask[0] > 0.5  # Threshold mask
            color = np.array(CLASS_COLORS[label])

            # Apply mask color
            for c in range(3):
                mask_overlay[:, :, c] = np.where(mask_binary,
                                                 color[c],
                                                 mask_overlay[:, :, c])

        # Blend with original image
        ax3.imshow(mask_overlay.astype(np.uint8), alpha=0.7)

    ax3.set_title('🎭 Segmentation Masks', fontsize=14, fontweight='bold')
    ax3.axis('off')

    # 4. Combined View (Boxes + Masks)
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.imshow(img_np)

    # Add masks
    if len(predictions['masks']) > 0:
        mask_overlay = np.zeros_like(img_np, dtype=np.float32)
        for mask, label in zip(predictions['masks'], predictions['labels']):
            mask_binary = mask[0] > 0.5
            color = np.array(CLASS_COLORS[label])
            for c in range(3):
                mask_overlay[:, :, c] = np.where(mask_binary, color[c], mask_overlay[:, :, c])
        ax4.imshow(mask_overlay.astype(np.uint8), alpha=0.5)

    # Add boxes
    for box, label, score in zip(predictions['boxes'], predictions['labels'], predictions['scores']):
        x1, y1, x2, y2 = box
        width, height = x2 - x1, y2 - y1
        color = 'red' if label == 1 else 'green'
        rect = patches.Rectangle((x1, y1), width, height, linewidth=2,
                                 edgecolor=color, facecolor='none')
        ax4.add_patch(rect)

    ax4.set_title('🔬 Combined View', fontsize=14, fontweight='bold')
    ax4.axis('off')

    # 5. Detailed Analysis (Bottom section)
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')

    # Create comprehensive analysis text
    analysis_text = create_detailed_analysis(predictions, ground_truth_info, detection_info)

    ax5.text(0.02, 0.98, analysis_text, transform=ax5.transAxes,
             fontsize=14, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=1", facecolor='lightblue', alpha=0.8))

    # Display the plot
    plt.show()

    # Keep the plot open and wait for user input
    input("\n🖱️  Press Enter to continue to next image...")
    plt.close(fig)

    return fig


def create_detailed_analysis(predictions, ground_truth_info, detection_info):
    """Create detailed analysis text"""
    analysis = "🔍 DETAILED DETECTION ANALYSIS\n"
    analysis += "=" * 80 + "\n\n"

    if len(predictions['labels']) == 0:
        analysis += "❌ NO DETECTIONS FOUND\n"
        analysis += f"   • Confidence threshold: {CONFIDENCE_THRESHOLD:.2f}\n"
        analysis += f"   • This typically indicates the image appears AUTHENTIC\n"
        analysis += f"   • No regions detected with sufficient confidence as fake or real\n\n"

        verdict = "🟢 LIKELY AUTHENTIC (No suspicious regions detected)"
        confidence_level = "Low suspicion"

    else:
        fake_count = sum(1 for label in predictions['labels'] if label == 1)
        real_count = sum(1 for label in predictions['labels'] if label == 2)

        analysis += f"📊 DETECTION STATISTICS:\n"
        analysis += f"   • Total detections: {len(predictions['labels'])}\n"
        analysis += f"   • Fake regions detected: {fake_count}\n"
        analysis += f"   • Real/Authentic regions: {real_count}\n\n"

        analysis += f"🎯 INDIVIDUAL DETECTIONS:\n"
        for i, info in enumerate(detection_info, 1):
            analysis += f"   {i}. {info}\n"
        analysis += "\n"

        # Calculate average confidences
        fake_scores = [score for score, label in zip(predictions['scores'], predictions['labels']) if label == 1]
        real_scores = [score for score, label in zip(predictions['scores'], predictions['labels']) if label == 2]

        if fake_scores:
            avg_fake_conf = np.mean(fake_scores)
            analysis += f"🔴 Average FAKE confidence: {avg_fake_conf:.3f}\n"

        if real_scores:
            avg_real_conf = np.mean(real_scores)
            analysis += f"🟢 Average REAL confidence: {avg_real_conf:.3f}\n"

        analysis += "\n"

        # Determine verdict
        if fake_count > real_count:
            verdict = "🚨 LIKELY DEEPFAKE DETECTED!"
            confidence_level = f"High confidence (avg: {np.mean(fake_scores):.3f})" if fake_scores else "Medium"
        elif real_count > fake_count:
            verdict = "✅ LIKELY AUTHENTIC"
            confidence_level = f"High confidence (avg: {np.mean(real_scores):.3f})" if real_scores else "Medium"
        else:
            verdict = "⚖️ MIXED RESULTS - REQUIRES MANUAL REVIEW"
            confidence_level = "Uncertain"

    analysis += f"🤖 FINAL VERDICT: {verdict}\n"
    analysis += f"📈 Confidence Level: {confidence_level}\n\n"

    # Add ground truth comparison if available
    if ground_truth_info:
        analysis += f"📋 GROUND TRUTH COMPARISON:\n"
        analysis += f"   • Actual label: {ground_truth_info.upper()}\n"

        # Check accuracy
        gt_lower = ground_truth_info.lower()
        if 'fake' in gt_lower and 'DEEPFAKE' in verdict:
            analysis += f"   • ✅ CORRECT DETECTION!\n"
        elif ('real' in gt_lower or 'authentic' in gt_lower) and (
                'AUTHENTIC' in verdict or len(predictions['labels']) == 0):
            analysis += f"   • ✅ CORRECT DETECTION!\n"
        else:
            analysis += f"   • ❓ REQUIRES FURTHER ANALYSIS\n"

    analysis += "\n" + "=" * 80

    return analysis


# ========= DEMO FUNCTIONS =========
def get_random_images(image_dir, ann_file, num_images=3):
    """Get random images from the dataset with ground truth info"""
    print(f"🎲 Loading annotations from: {ann_file}")
    coco = COCO(ann_file)
    img_ids = list(coco.imgs.keys())
    print(f"📁 Found {len(img_ids)} images in dataset")

    # Randomly sample images
    selected_ids = random.sample(img_ids, min(num_images, len(img_ids)))

    selected_images = []
    for img_id in selected_ids:
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(image_dir, img_info['file_name'])

        if os.path.exists(img_path):
            # Get ground truth annotations
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)

            # Determine ground truth label
            gt_labels = []
            for ann in anns:
                if not ann.get('iscrowd', 0):
                    category_id = ann.get('category_id', 1)
                    gt_label = 'fake' if category_id == 1 else 'real'
                    gt_labels.append(gt_label)

            gt_summary = ', '.join(set(gt_labels)) if gt_labels else 'unknown'

            selected_images.append({
                'id': img_id,
                'path': img_path,
                'filename': img_info['file_name'],
                'ground_truth': gt_summary
            })

    return selected_images


def run_interactive_demo(model, selected_images):
    """Run the interactive demo with real-time display"""
    print(f"\n🎬 Starting Interactive Demo with {len(selected_images)} images")
    print("🖥️  Images will be displayed in PyCharm's plot window")
    print("=" * 80)

    results = []

    for i, img_data in enumerate(selected_images, 1):
        print(f"\n📸 Processing Image {i}/{len(selected_images)}: {img_data['filename']}")
        print(f"🏷️  Ground Truth: {img_data['ground_truth']}")
        print("🔄 Loading and analyzing...")

        try:
            # Load and preprocess image
            original_img, resized_img, tensor, original_size = preprocess_image(img_data['path'])
            print(f"✅ Image loaded: {original_size[0]}x{original_size[1]} -> {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}")

            # Get predictions
            print("🤖 Running model prediction...")
            predictions = predict_image(model, tensor, DEVICE, CONFIDENCE_THRESHOLD)

            # Quick summary
            fake_count = sum(1 for label in predictions['labels'] if label == 1)
            real_count = sum(1 for label in predictions['labels'] if label == 2)

            print(f"📊 Quick Results: {len(predictions['labels'])} detections ({fake_count} fake, {real_count} real)")

            # Display interactive results
            print("🖼️  Displaying results... (check PyCharm plot window)")
            display_interactive_results(
                original_img,
                predictions,
                img_data['ground_truth'],
                img_data['filename']
            )

            # Store results
            if len(predictions['labels']) == 0:
                verdict = "No detections (likely authentic)"
                confidence = 0.0
            elif fake_count > real_count:
                verdict = "LIKELY DEEPFAKE"
                confidence = np.mean(
                    [score for score, label in zip(predictions['scores'], predictions['labels']) if label == 1])
            elif real_count > fake_count:
                verdict = "LIKELY AUTHENTIC"
                confidence = np.mean(
                    [score for score, label in zip(predictions['scores'], predictions['labels']) if label == 2])
            else:
                verdict = "MIXED/UNCERTAIN"
                confidence = np.mean(predictions['scores'])

            results.append({
                'filename': img_data['filename'],
                'ground_truth': img_data['ground_truth'],
                'verdict': verdict,
                'confidence': confidence,
                'detections': len(predictions['labels']),
                'fake_count': fake_count,
                'real_count': real_count
            })

            print(f"✅ Image {i} analysis complete!")

        except Exception as e:
            print(f"❌ Error processing {img_data['filename']}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    return results


def print_final_summary(results):
    """Print final summary of all results"""
    print("\n" + "🎉" * 30)
    print("INTERACTIVE DEMO COMPLETE!")
    print("🎉" * 30)

    print(f"\n📈 PERFORMANCE SUMMARY:")
    print("=" * 50)

    correct_predictions = 0
    total_predictions = len(results)

    for i, result in enumerate(results, 1):
        print(f"\n📸 Image {i}: {result['filename']}")
        print(f"   🏷️  Ground Truth: {result['ground_truth']}")
        print(f"   🤖 Model Verdict: {result['verdict']}")
        print(f"   📊 Confidence: {result['confidence']:.3f}")
        print(
            f"   🔍 Detections: {result['detections']} total ({result['fake_count']} fake, {result['real_count']} real)")

        # Check accuracy
        gt_lower = result['ground_truth'].lower()
        verdict_lower = result['verdict'].lower()

        is_correct = False
        if 'fake' in gt_lower and 'deepfake' in verdict_lower:
            is_correct = True
        elif 'real' in gt_lower and ('authentic' in verdict_lower or result['detections'] == 0):
            is_correct = True

        if is_correct:
            print("   ✅ CORRECT PREDICTION!")
            correct_predictions += 1
        else:
            print("   ❌ INCORRECT/UNCERTAIN")

    print(
        f"\n📊 OVERALL ACCURACY: {correct_predictions}/{total_predictions} ({100 * correct_predictions / total_predictions:.1f}%)")
    print("\n🎯 Demo completed successfully!")


# ========= MAIN DEMO =========
def \
        main():
    """Main interactive demo function"""
    print("🚀 INTERACTIVE DEEPFAKE DETECTION DEMO")
    print("=" * 60)
    print("🖥️  This demo will show detection results in PyCharm's plot window")
    print("🖱️  You can interact with the plots and see detailed analysis")
    print("=" * 60)

    print(f"📦 Model: {MODEL_PATH}")
    print(f"🖥️  Device: {DEVICE}")
    print(f"🎯 Confidence Threshold: {CONFIDENCE_THRESHOLD}")
    print(f"📁 Dataset: {CHALLENGE_CONFIG['image_dir']}")

    # Validation checks
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found: {MODEL_PATH}")
        return

    if not os.path.exists(CHALLENGE_CONFIG['image_dir']) or not os.path.exists(CHALLENGE_CONFIG['ann_file']):
        print(f"❌ Dataset files not found!")
        print(f"   Image directory: {CHALLENGE_CONFIG['image_dir']}")
        print(f"   Annotation file: {CHALLENGE_CONFIG['ann_file']}")
        return

    # Load model
    model = load_model(MODEL_PATH, DEVICE)

    # Get random images
    print(f"\n🎲 Selecting 3 random images from challenge dataset...")
    selected_images = get_random_images(
        CHALLENGE_CONFIG['image_dir'],
        CHALLENGE_CONFIG['ann_file'],
        num_images=3
    )

    if not selected_images:
        print("❌ No valid images found in dataset!")
        return

    print(f"✅ Selected {len(selected_images)} images for interactive demo")
    for i, img in enumerate(selected_images, 1):
        print(f"   {i}. {img['filename']} (GT: {img['ground_truth']})")

    input("\n🚀 Press Enter to start the interactive demo...")

    # Run interactive demo
    results = run_interactive_demo(model, selected_images)

    # Print final summary
    print_final_summary(results)


if __name__ == "__main__":
    # Set random seed for reproducible demo
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    main()