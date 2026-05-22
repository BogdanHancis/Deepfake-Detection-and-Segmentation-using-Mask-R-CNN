import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from pycocotools.coco import COCO
from torchvision.transforms import functional as F
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from collections import Counter
import pandas as pd

# ========= CONFIGURATION =========
MODEL_PATH = "C:/Users/bogdan/PycharmProjects/Licence/best_maskrcnn_iou800.pth"  # Your trained model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = (256, 256)
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
OUTPUT_DIR = r"C:\Users\bogdan\OneDrive\Desktop\Licenta-Chapter3\results 1\evaluation_results"

# Class mapping - focusing only on fake vs real
CLASS_NAMES = {1: 'fake', 2: 'real'}
BINARY_CLASS_NAMES = {0: 'fake', 1: 'real'}  # For binary confusion matrix

# Test datasets
TEST_DATASETS = {
    "Test-Dev": {
        "image_dir": "D:/Deepfake/datasets/Test-Dev",
        "ann_file": "D:/Deepfake/datasets/Annotations/Test-Dev_poly_cleaned.json"
    },
    "Test-Challenge": {
        "image_dir": "D:/Deepfake/datasets/Test-Challenge",
        "ann_file": "D:/Deepfake/datasets/Annotations/Test-Challenge_poly_cleaned.json"
    },
    "Validation": {
        "image_dir": "D:/Deepfake/datasets/Val",
        "ann_file": "D:/Deepfake/datasets/Annotations/Val_poly_cleaned.json"
    }
}


# ========= MODEL SETUP =========
def create_model(num_classes=3):
    """Create the same model architecture used in training"""
    model = maskrcnn_resnet50_fpn(weights="DEFAULT")

    # Replace the classifier head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Replace the mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)

    return model


def load_trained_model(model_path, device):
    """Load your trained model"""
    print(f"Loading model from: {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = create_model(num_classes=3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    print("✅ Model loaded successfully")
    return model


# ========= EVALUATION FUNCTIONS =========
def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Calculate intersection
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)

    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0

    intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def match_predictions_to_gt_binary(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels):
    """Match predictions to ground truth - focusing only on fake vs real matches"""
    matched_pairs = []

    # Filter predictions by confidence and remove background predictions
    valid_preds = (pred_scores >= CONFIDENCE_THRESHOLD) & (pred_labels > 0)
    if not valid_preds.any():
        # If no valid predictions, all GT become false negatives
        for gt_label in gt_labels:
            matched_pairs.append({
                'pred_label': None,  # No prediction
                'gt_label': gt_label,
                'confidence': 0.0,
                'iou': 0.0,
                'correct': False,
                'match_type': 'false_negative'
            })
        return matched_pairs

    pred_boxes = pred_boxes[valid_preds]
    pred_labels = pred_labels[valid_preds]
    pred_scores = pred_scores[valid_preds]

    # Sort by confidence (highest first)
    sorted_indices = torch.argsort(pred_scores, descending=True)
    used_gt = set()

    for pred_idx in sorted_indices:
        pred_box = pred_boxes[pred_idx].cpu().numpy()
        pred_label = pred_labels[pred_idx].item()
        pred_score = pred_scores[pred_idx].item()

        best_iou = 0.0
        best_gt_idx = -1

        # Find best matching ground truth
        for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
            if gt_idx in used_gt:
                continue

            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou and iou >= IOU_THRESHOLD:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_gt_idx != -1:
            # True positive or false positive (based on class match)
            gt_label = gt_labels[best_gt_idx]
            matched_pairs.append({
                'pred_label': pred_label,
                'gt_label': gt_label,
                'confidence': pred_score,
                'iou': best_iou,
                'correct': pred_label == gt_label,
                'match_type': 'true_positive' if pred_label == gt_label else 'false_positive_class'
            })
            used_gt.add(best_gt_idx)
        else:
            # False positive - predicted something but no good GT match
            # We'll skip these for binary classification as they don't have a clear GT class
            pass

    # Add false negatives (GT that wasn't matched)
    for gt_idx, gt_label in enumerate(gt_labels):
        if gt_idx not in used_gt:
            matched_pairs.append({
                'pred_label': None,  # No prediction
                'gt_label': gt_label,
                'confidence': 0.0,
                'iou': 0.0,
                'correct': False,
                'match_type': 'false_negative'
            })

    return matched_pairs


def evaluate_dataset_binary(model, image_dir, ann_file, dataset_name):
    """Evaluate model on a single dataset - binary classification focus"""
    print(f"\n📊 Evaluating {dataset_name} dataset (Fake vs Real only)...")

    # Load COCO annotations
    coco_gt = COCO(ann_file)
    img_ids = list(coco_gt.imgs.keys())

    all_results = []

    for img_id in tqdm(img_ids, desc=f"Processing {dataset_name}"):
        try:
            # Load image
            img_info = coco_gt.loadImgs(img_id)[0]
            img_path = os.path.join(image_dir, img_info['file_name'])

            if not os.path.exists(img_path):
                continue

            # Prepare image
            img = Image.open(img_path).convert("RGB")
            img_resized = img.resize(IMAGE_SIZE)
            img_tensor = F.to_tensor(img_resized).unsqueeze(0).to(DEVICE)

            # Get predictions
            with torch.no_grad():
                predictions = model(img_tensor)[0]

            # Get ground truth
            ann_ids = coco_gt.getAnnIds(imgIds=img_id)
            anns = coco_gt.loadAnns(ann_ids)

            # Process ground truth
            gt_boxes = []
            gt_labels = []

            for ann in anns:
                if ann.get('iscrowd', 0):
                    continue

                # Scale bbox to resized image
                bbox = ann['bbox']
                scale_x = IMAGE_SIZE[0] / img_info['width']
                scale_y = IMAGE_SIZE[1] / img_info['height']

                x, y, w, h = bbox
                x = x * scale_x
                y = y * scale_y
                w = w * scale_x
                h = h * scale_y

                gt_boxes.append([x, y, x + w, y + h])

                # Convert category to label (1=fake, 2=real)
                category_id = ann.get('category_id', 1)
                label = 1 if category_id == 1 else 2
                gt_labels.append(label)

            if not gt_boxes:
                continue

            # Match predictions to ground truth
            matches = match_predictions_to_gt_binary(
                predictions['boxes'],
                predictions['labels'],
                predictions['scores'],
                gt_boxes,
                gt_labels
            )

            # Store results
            for match in matches:
                if match['pred_label'] is not None or match['gt_label'] in [1, 2]:
                    match['dataset'] = dataset_name
                    match['image_id'] = img_id
                    all_results.append(match)

        except Exception as e:
            print(f"⚠️ Error processing image {img_id}: {e}")
            continue

    return all_results


# ========= VISUALIZATION FUNCTIONS =========
def plot_binary_confusion_matrix(y_true, y_pred, dataset_name, save_path):
    """Plot binary confusion matrix for fake vs real"""
    # Convert to binary (0=fake, 1=real)
    y_true_binary = [0 if label == 1 else 1 for label in y_true]
    y_pred_binary = [0 if label == 1 else 1 for label in y_pred]

    cm = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1])

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Fake', 'Real'],
                yticklabels=['Fake', 'Real'])

    plt.title(f'Binary Confusion Matrix - {dataset_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_all_binary_confusion_matrices(results_dict, output_dir):
    """Plot binary confusion matrices for all datasets"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    # Individual datasets
    for idx, (dataset_name, results) in enumerate(results_dict.items()):
        if idx >= 3:  # Save space for combined
            break

        # Filter out None predictions and convert to binary
        valid_results = [r for r in results if r['pred_label'] is not None]
        if not valid_results:
            continue

        y_true = [r['gt_label'] for r in valid_results]
        y_pred = [r['pred_label'] for r in valid_results]

        # Convert to binary (0=fake, 1=real)
        y_true_binary = [0 if label == 1 else 1 for label in y_true]
        y_pred_binary = [0 if label == 1 else 1 for label in y_pred]

        cm = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1])

        ax = axes[idx]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Fake', 'Real'],
                    yticklabels=['Fake', 'Real'], ax=ax)
        ax.set_title(f'{dataset_name}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')

    # Combined results
    all_true = []
    all_pred = []
    for results in results_dict.values():
        valid_results = [r for r in results if r['pred_label'] is not None]
        all_true.extend([r['gt_label'] for r in valid_results])
        all_pred.extend([r['pred_label'] for r in valid_results])

    # Convert to binary
    all_true_binary = [0 if label == 1 else 1 for label in all_true]
    all_pred_binary = [0 if label == 1 else 1 for label in all_pred]

    cm_combined = confusion_matrix(all_true_binary, all_pred_binary, labels=[0, 1])
    ax = axes[-1]
    sns.heatmap(cm_combined, annot=True, fmt='d', cmap='Reds',
                xticklabels=['Fake', 'Real'],
                yticklabels=['Fake', 'Real'], ax=ax)
    ax.set_title('Combined Results')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'binary_confusion_matrices.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


def calculate_binary_metrics(y_true, y_pred):
    """Calculate binary classification metrics"""
    # Convert to binary (0=fake, 1=real)
    y_true_binary = [0 if label == 1 else 1 for label in y_true]
    y_pred_binary = [0 if label == 1 else 1 for label in y_pred]

    accuracy = accuracy_score(y_true_binary, y_pred_binary)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_binary, y_pred_binary, labels=[0, 1], average=None, zero_division=0
    )

    # Macro averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true_binary, y_pred_binary, average='macro', zero_division=0
    )

    # Weighted averages
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true_binary, y_pred_binary, average='weighted', zero_division=0
    )

    return {
        'accuracy': accuracy,
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'f1_score': f1.tolist(),
        'support': support.tolist(),
        'macro_precision': precision_macro,
        'macro_recall': recall_macro,
        'macro_f1': f1_macro,
        'weighted_precision': precision_weighted,
        'weighted_recall': recall_weighted,
        'weighted_f1': f1_weighted
    }


# ========= MAIN EVALUATION =========
def main():
    """Main evaluation function - Binary Classification Focus"""
    print("🚀 Starting Binary Confusion Matrix Evaluation (Fake vs Real)")
    print(f"📦 Model: {MODEL_PATH}")
    print(f"🖥️  Device: {DEVICE}")
    print(f"📏 Image Size: {IMAGE_SIZE}")
    print(f"🎯 Confidence Threshold: {CONFIDENCE_THRESHOLD}")
    print(f"📊 IoU Threshold: {IOU_THRESHOLD}")
    print(f"📁 Output Directory: {OUTPUT_DIR}")

    # Create output directory
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"✅ Output directory created/verified: {OUTPUT_DIR}")
    except Exception as e:
        print(f"❌ Error creating output directory: {e}")
        return

    # Load model
    try:
        model = load_trained_model(MODEL_PATH, DEVICE)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return

    # Evaluate each dataset
    all_results = {}

    for dataset_name, config in TEST_DATASETS.items():
        if not os.path.exists(config['image_dir']) or not os.path.exists(config['ann_file']):
            print(f"⚠️ Skipping {dataset_name} - missing files")
            continue

        results = evaluate_dataset_binary(
            model,
            config['image_dir'],
            config['ann_file'],
            dataset_name
        )

        if results:
            all_results[dataset_name] = results

    if not all_results:
        print("❌ No datasets were successfully evaluated!")
        return

    # Generate visualizations
    print("\n📊 Generating binary confusion matrices...")
    plot_all_binary_confusion_matrices(all_results, OUTPUT_DIR)

    # Calculate and display metrics
    print("\n" + "=" * 80)
    print("📈 BINARY CLASSIFICATION RESULTS (FAKE vs REAL)")
    print("=" * 80)

    summary_data = []

    for dataset_name, results in all_results.items():
        # Filter valid predictions only
        valid_results = [r for r in results if r['pred_label'] is not None]
        total_gt = len(results)  # Include false negatives
        detected = len(valid_results)

        if not valid_results:
            print(f"\n📊 {dataset_name} Results: No valid detections!")
            continue

        y_true = [r['gt_label'] for r in valid_results]
        y_pred = [r['pred_label'] for r in valid_results]

        # Calculate metrics
        metrics = calculate_binary_metrics(y_true, y_pred)

        # Display results
        print(f"\n📊 {dataset_name} Results:")
        print(f"  Total GT Objects: {total_gt}")
        print(f"  Detected Objects: {detected}")
        print(f"  Detection Rate: {detected / total_gt:.3f}")
        print(f"  Binary Accuracy: {metrics['accuracy']:.3f}")
        print(f"  Macro F1: {metrics['macro_f1']:.3f}")
        print(f"  Weighted F1: {metrics['weighted_f1']:.3f}")

        print("  Per-class metrics:")
        for i, class_name in enumerate(['Fake', 'Real']):
            print(f"    {class_name:>6}: P={metrics['precision'][i]:.3f} "
                  f"R={metrics['recall'][i]:.3f} F1={metrics['f1_score'][i]:.3f} "
                  f"Support={metrics['support'][i]}")

        # Show binary confusion matrix
        y_true_binary = [0 if label == 1 else 1 for label in y_true]
        y_pred_binary = [0 if label == 1 else 1 for label in y_pred]
        cm = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1])
        print(f"  Binary Confusion Matrix:")
        print(f"    True\\Pred  {'Fake':>6} {'Real':>6}")
        print(f"    {'Fake':>8}  {cm[0, 0]:>6} {cm[0, 1]:>6}")
        print(f"    {'Real':>8}  {cm[1, 0]:>6} {cm[1, 1]:>6}")

        # Store for CSV
        summary_data.append({
            'Dataset': dataset_name,
            'Total_GT': total_gt,
            'Detected': detected,
            'Detection_Rate': detected / total_gt,
            'Binary_Accuracy': metrics['accuracy'],
            'Macro_F1': metrics['macro_f1'],
            'Weighted_F1': metrics['weighted_f1'],
            'Fake_F1': metrics['f1_score'][0],
            'Real_F1': metrics['f1_score'][1],
            'Fake_Precision': metrics['precision'][0],
            'Real_Precision': metrics['precision'][1],
            'Fake_Recall': metrics['recall'][0],
            'Real_Recall': metrics['recall'][1]
        })

    # Combined results
    print(f"\n🎯 COMBINED BINARY RESULTS:")
    all_valid_results = []
    all_total = 0

    for results in all_results.values():
        all_total += len(results)
        all_valid_results.extend([r for r in results if r['pred_label'] is not None])

    if all_valid_results:
        all_true = [r['gt_label'] for r in all_valid_results]
        all_pred = [r['pred_label'] for r in all_valid_results]

        combined_metrics = calculate_binary_metrics(all_true, all_pred)
        print(f"  Total GT Objects: {all_total}")
        print(f"  Total Detected: {len(all_valid_results)}")
        print(f"  Overall Detection Rate: {len(all_valid_results) / all_total:.3f}")
        print(f"  Overall Binary Accuracy: {combined_metrics['accuracy']:.3f}")
        print(f"  Overall Macro F1: {combined_metrics['macro_f1']:.3f}")
        print(f"  Overall Weighted F1: {combined_metrics['weighted_f1']:.3f}")

        # Show combined binary confusion matrix
        all_true_binary = [0 if label == 1 else 1 for label in all_true]
        all_pred_binary = [0 if label == 1 else 1 for label in all_pred]
        cm_combined = confusion_matrix(all_true_binary, all_pred_binary, labels=[0, 1])
        print(f"  Combined Binary Confusion Matrix:")
        print(f"    True\\Pred  {'Fake':>6} {'Real':>6}")
        print(f"    {'Fake':>8}  {cm_combined[0, 0]:>6} {cm_combined[0, 1]:>6}")
        print(f"    {'Real':>8}  {cm_combined[1, 0]:>6} {cm_combined[1, 1]:>6}")

        # Add to summary
        summary_data.append({
            'Dataset': 'COMBINED',
            'Total_GT': all_total,
            'Detected': len(all_valid_results),
            'Detection_Rate': len(all_valid_results) / all_total,
            'Binary_Accuracy': combined_metrics['accuracy'],
            'Macro_F1': combined_metrics['macro_f1'],
            'Weighted_F1': combined_metrics['weighted_f1'],
            'Fake_F1': combined_metrics['f1_score'][0],
            'Real_F1': combined_metrics['f1_score'][1],
            'Fake_Precision': combined_metrics['precision'][0],
            'Real_Precision': combined_metrics['precision'][1],
            'Fake_Recall': combined_metrics['recall'][0],
            'Real_Recall': combined_metrics['recall'][1]
        })

    # Save CSV
    if summary_data:
        df = pd.DataFrame(summary_data)
        csv_path = os.path.join(OUTPUT_DIR, 'binary_evaluation_summary.csv')
        df.to_csv(csv_path, index=False)

        # Show final class distribution
        print(f"\n📊 Final Class Distribution (Detected Objects Only):")
        if all_valid_results:
            true_counts = Counter([r['gt_label'] for r in all_valid_results])
            pred_counts = Counter([r['pred_label'] for r in all_valid_results])

            for class_id, class_name in CLASS_NAMES.items():
                true_count = true_counts.get(class_id, 0)
                pred_count = pred_counts.get(class_id, 0)
                print(f"  {class_name:>6}: True={true_count:>5} Pred={pred_count:>5}")

        print(f"\n📁 Results saved to: {OUTPUT_DIR}")
        print(f"📊 Summary CSV: {csv_path}")
        print(f"📈 Binary confusion matrices: {OUTPUT_DIR}/binary_confusion_matrices.png")
        print("\n✅ Binary evaluation completed successfully!")


if __name__ == "__main__":
    main()