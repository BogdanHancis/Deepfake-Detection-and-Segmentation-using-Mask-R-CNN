__author__ = 'ltnghia'

import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from cocoeval import COCOeval
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report


def visualize_comparison(cocoGt, cocoDt, img_id, image_dir):
    """Visualize ground truth vs predictions for debugging"""
    # Load image
    img_info = cocoGt.loadImgs(img_id)[0]
    img_path = os.path.join(image_dir, img_info['file_name'])
    image = Image.open(img_path).convert("RGB")

    # Create figure
    plt.figure(figsize=(15, 8))

    # Ground truth
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    ann_ids = cocoGt.getAnnIds(imgIds=img_id)
    anns = cocoGt.loadAnns(ann_ids)
    for ann in anns:
        bbox = ann['bbox']
        rect = plt.Rectangle(
            (bbox[0], bbox[1]), bbox[2], bbox[3],
            fill=False, color='green', linewidth=2
        )
        plt.gca().add_patch(rect)
        plt.text(
            bbox[0], bbox[1], f"GT: {cocoGt.loadCats(ann['category_id'])[0]['name']}",
            color='white', bbox=dict(facecolor='green', alpha=0.5)
        )
    plt.title('Ground Truth')
    plt.axis('off')

    # Predictions
    plt.subplot(1, 2, 2)
    plt.imshow(image)
    pred_ids = cocoDt.getAnnIds(imgIds=img_id)
    preds = cocoDt.loadAnns(pred_ids)
    for pred in preds:
        bbox = pred['bbox']
        rect = plt.Rectangle(
            (bbox[0], bbox[1]), bbox[2], bbox[3],
            fill=False, color='red', linewidth=2
        )
        plt.gca().add_patch(rect)
        try:
            cat_name = cocoDt.loadCats(pred['category_id'])[0]['name']
        except KeyError:
            cat_name = f"Unknown category ({pred['category_id']})"
        plt.text(
            bbox[0], bbox[1], f"Pred: {cat_name} {pred['score']:.2f}",
            color='white', bbox=dict(facecolor='red', alpha=0.5)
        )
    plt.title('Predictions')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(f"eval_debug_{img_id}.png", bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved visualization to eval_debug_{img_id}.png")


def evaluate_classification(cocoGt, cocoDt):
    """Evaluate predictions as classification problem"""
    print("\n" + "=" * 60)
    print("🔍 Evaluating as Classification Problem")
    print("=" * 60)

    img_ids = cocoGt.getImgIds()
    predicted_classes = []
    true_classes = []

    # Get all category IDs and names
    cat_ids = cocoGt.getCatIds()
    cat_names = [cocoGt.loadCats(cat_id)[0]['name'] for cat_id in cat_ids]

    for img_id in img_ids:
        # Get ground truth classes for this image
        ann_ids = cocoGt.getAnnIds(imgIds=img_id)
        anns = cocoGt.loadAnns(ann_ids)

        # Get predictions for this image
        preds = cocoDt.loadAnns(cocoDt.getAnnIds(imgIds=img_id))

        # Only evaluate if we have both ground truth and predictions
        if anns and preds:
            # For simplicity, take the first annotation's category as ground truth
            true_class = anns[0]['category_id']
            true_classes.append(true_class)

            # Take the highest confidence prediction
            best_pred = max(preds, key=lambda x: x['score'])
            # Check if predicted category exists in ground truth
            if best_pred['category_id'] in cat_ids:
                predicted_classes.append(best_pred['category_id'])
            else:
                # Skip predictions with invalid categories
                continue

    # Verify we have matching pairs
    if len(true_classes) != len(predicted_classes):
        print(f"⚠️ Mismatch in sample counts: {len(true_classes)} GT vs {len(predicted_classes)} preds")
        print("This suggests some images had predictions but no GT or vice versa")

        # Find the minimum length to compare
        min_length = min(len(true_classes), len(predicted_classes))
        true_classes = true_classes[:min_length]
        predicted_classes = predicted_classes[:min_length]

    if len(true_classes) > 0 and len(predicted_classes) > 0:
        print(f"\nEvaluating on {len(true_classes)} matched image samples")

        accuracy = accuracy_score(true_classes, predicted_classes)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_classes, predicted_classes, average='weighted', zero_division=0
        )

        print(f"\nClassification Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision (weighted): {precision:.4f}")
        print(f"Recall (weighted): {recall:.4f}")
        print(f"F1 Score (weighted): {f1:.4f}")

        # Confusion matrix
        print("\nConfusion Matrix:")
        cm = confusion_matrix(true_classes, predicted_classes, labels=cat_ids)
        print(cm)

        # Classification report
        print("\nClassification Report:")
        print(classification_report(
            true_classes, predicted_classes,
            target_names=cat_names,
            zero_division=0
        ))
    else:
        print("\n⚠️ No valid prediction-ground truth pairs found for classification evaluation")

if __name__ == '__main__':
    # ========= CONFIG =========
    gt_file = 'D:/Deepfake/datasets/Annotations/Test-Dev_poly_cleaned.json'
    result_file = 'C:/Users/bogdan/PycharmProjects/Licence/predictions/pred_test_dev.json'
    image_dir = 'D:/Deepfake/datasets/Test-Dev'  # Path to your images
    tasks = ['segm']
    debug_samples = 3  # Number of random samples to visualize

    # ========= CHECK FILES =========
    if not os.path.exists(gt_file):
        print(f"❌ Ground truth file not found: {gt_file}")
        exit(1)
    if not os.path.exists(result_file):
        print(f"❌ Prediction result file not found: {result_file}")
        exit(1)

    # ========= LOAD DATA =========
    print(f"\n📂 Loading GT from: {gt_file}")
    cocoGt = COCO(gt_file)

    print("\nGround Truth Categories:")
    for cat_id, cat_info in cocoGt.cats.items():
        print(f"  ID: {cat_id}, Name: {cat_info['name']}")

    print(f"\n📂 Loading predictions from: {result_file}")
    with open(result_file) as f:
        pred_data = json.load(f)

    if 'predictions' in pred_data:
        predictions = pred_data['predictions']
        print("\nPrediction Metadata:")
        print(f"  Model: {pred_data['info']['model']}")
        print(f"  Score Threshold: {pred_data['info']['score_threshold']}")
    else:
        predictions = pred_data

    # Fix prediction categories to match ground truth
    # Get the first category ID from ground truth as default
    default_cat_id = cocoGt.getCatIds()[0]
    for pred in predictions:
        if 'category_id' not in pred or pred['category_id'] not in cocoGt.getCatIds():
            print(
                f"⚠️ Prediction has invalid category_id: {pred.get('category_id', 'missing')}, mapping to {default_cat_id}")
            pred['category_id'] = default_cat_id

    cocoDt = cocoGt.loadRes(predictions)

    # ========= DEBUG VISUALIZATION =========
    print(f"\n🔍 Generating {debug_samples} debug visualizations...")
    img_ids = cocoGt.getImgIds()
    for _ in range(min(debug_samples, len(img_ids))):
        img_id = random.choice(img_ids)

        # Print annotation details
        print(f"\nImage ID: {img_id}")
        print("Ground Truth Annotations:")
        for ann in cocoGt.loadAnns(cocoGt.getAnnIds(imgIds=img_id)):
            print(f"  - Category: {cocoGt.loadCats(ann['category_id'])[0]['name']}")
            print(f"    BBox: {ann['bbox']}")
            print(f"    Area: {ann['area']}")

        print("\nPredicted Annotations:")
        for pred in cocoDt.loadAnns(cocoDt.getAnnIds(imgIds=img_id)):
            try:
                cat_name = cocoDt.loadCats(pred['category_id'])[0]['name']
            except KeyError:
                cat_name = f"Unknown category ({pred['category_id']})"
            print(f"  - Category: {cat_name}")
            print(f"    Score: {pred['score']:.2f}")
            print(f"    BBox: {pred['bbox']}")

        # Generate visualization
        visualize_comparison(cocoGt, cocoDt, img_id, image_dir)

    # ========= RUN EVALUATION =========
    # Standard COCO evaluation
    for task in tasks:
        print("\n" + "=" * 60)
        print(f"🔍 Evaluating task: {task.upper()}")
        print("=" * 60)

        cocoEval = COCOeval(cocoGt, cocoDt, iouType=task)
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        # Print per-category results
        print("\nPer-category results:")
        for cat_id in cocoGt.getCatIds():
            stats = cocoEval.eval['precision'][:, :, cat_id - 1, 0, 2]  # AP@[IoU=0.5:0.95]
            print(f"{cocoGt.loadCats(cat_id)[0]['name']}: {stats.mean():.3f}")

    # Classification evaluation
    evaluate_classification(cocoGt, cocoDt)

    print("\n✅ Evaluation completed.")