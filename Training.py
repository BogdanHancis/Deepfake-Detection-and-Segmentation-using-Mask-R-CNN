import os
import json
import argparse
import random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
from pycocotools.coco import COCO
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from PIL import Image
from tqdm import trange
import matplotlib.pyplot as plt
import numpy as np
import time
from datetime import datetime

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
scaler = torch.cuda.amp.GradScaler()


class CocoDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, ann_file, resize_size=(256, 256), max_samples=None):
        self.coco = COCO(ann_file)
        self.image_dir = image_dir
        self.resize_size = resize_size

        self.ids = list(sorted(self.coco.imgs.keys()))
        self.valid_ids = []

        print(f"🔍 Verifying images in {image_dir}...")
        for img_id in self.ids:
            img_info = self.coco.loadImgs(img_id)[0]
            img_path = os.path.join(self.image_dir, img_info['file_name'])
            if os.path.exists(img_path):
                self.valid_ids.append(img_id)

        if max_samples:
            print(f"📉 Subsampling {max_samples} images out of {len(self.valid_ids)}")
            self.valid_ids = random.sample(self.valid_ids, min(max_samples, len(self.valid_ids)))

        print(f"✅ Using {len(self.valid_ids)} images for training.")
    def __getitem__(self, idx):
        try:
            img_id = self.valid_ids[idx]
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            image_info = self.coco.loadImgs(img_id)[0]

            img_path = os.path.join(self.image_dir, image_info['file_name'])
            img = Image.open(img_path).convert("RGB")
            img = img.resize(self.resize_size)

            masks, boxes, labels = [], [], []

            for ann in anns:
                m = self.coco.annToMask(ann)
                m = Image.fromarray(m).resize(self.resize_size, resample=Image.NEAREST)
                m = torch.tensor(np.array(m), dtype=torch.uint8)
                if m.sum() == 0:
                    continue
                masks.append(m)

                bbox = ann['bbox']
                scale_x = self.resize_size[0] / image_info['width']
                scale_y = self.resize_size[1] / image_info['height']
                x, y, w, h = bbox
                x, w = x * scale_x, w * scale_x
                y, h = y * scale_y, h * scale_y
                boxes.append([x, y, x + w, y + h])

                # Modified for 3 classes: 0-background, 1-fake, 2-real
                # Assuming category_id in annotation determines fake (1) or real (2)
                category_id = ann.get('category_id', 1)
                if category_id == 1:
                    labels.append(1)  # fake
                else:
                    labels.append(2)  # real

            if not boxes:
                return None

            boxes = torch.tensor(boxes, dtype=torch.float32)
            valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            if valid.sum() == 0:
                return None

            boxes = boxes[valid]
            masks = torch.stack(masks)[valid]
            labels = [labels[i] for i, v in enumerate(valid) if v]

            target = {
                "boxes": boxes,
                "labels": torch.tensor(labels, dtype=torch.int64),
                "masks": masks,
                "image_id": torch.tensor([img_id]),
            }

            return F.to_tensor(img), target

        except Exception as e:
            print(f"⚠️ Skipping sample idx {idx} due to error: {e}")
            return None

    def __len__(self):
        return len(self.valid_ids)


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return [], []
    return tuple(zip(*batch))


def get_model(num_classes):
    model = maskrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    return model


def calculate_iou(pred_masks, target_masks, threshold=0.5):
    """Calculate IoU between predicted and target masks"""
    if len(pred_masks) == 0 or len(target_masks) == 0:
        return 0.0

    ious = []
    pred_masks = (pred_masks > threshold).float()
    target_masks = target_masks.float()

    for pred_mask in pred_masks:
        best_iou = 0.0
        for target_mask in target_masks:
            intersection = (pred_mask * target_mask).sum()
            union = pred_mask.sum() + target_mask.sum() - intersection
            if union > 0:
                iou = intersection / union
                best_iou = max(best_iou, iou.item())
        ious.append(best_iou)

    return np.mean(ious) if ious else 0.0


def evaluate_model(model, val_loader, device):
    """Comprehensive model evaluation"""
    model.eval()
    total_iou = 0.0
    total_samples = 0
    class_correct = {1: 0, 2: 0}  # fake, real
    class_total = {1: 0, 2: 0}

    with torch.no_grad():
        for val_images, val_targets in val_loader:
            if len(val_images) == 0:
                continue

            val_images = [img.to(device) for img in val_images]
            val_targets = [{k: v.to(device) for k, v in t.items()} for t in val_targets]

            outputs = model(val_images)

            for output, target in zip(outputs, val_targets):
                if len(output['masks']) > 0 and len(target['masks']) > 0:
                    # Calculate IoU
                    iou = calculate_iou(
                        output['masks'][:, 0],
                        target['masks'],
                        threshold=0.5
                    )
                    total_iou += iou
                    total_samples += 1

                    # Calculate class accuracy
                    pred_labels = output['labels'][output['scores'] > 0.5]
                    target_labels = target['labels']

                    for label in target_labels:
                        label_val = label.item()
                        if label_val in class_total:
                            class_total[label_val] += 1
                            if label_val in pred_labels:
                                class_correct[label_val] += 1

    avg_iou = total_iou / total_samples if total_samples > 0 else 0.0
    class_accuracies = {
        k: class_correct[k] / class_total[k] if class_total[k] > 0 else 0.0
        for k in class_total
    }

    return avg_iou, class_accuracies


def save_checkpoint(model, optimizer, epoch, train_losses, val_losses, train_ious, val_ious,
                    train_class_acc, val_class_acc, checkpoint_path):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_ious': train_ious,
        'val_ious': val_ious,
        'train_class_accuracies': train_class_acc,
        'val_class_accuracies': val_class_acc,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"💾 Checkpoint saved: {checkpoint_path}")


def load_checkpoint(checkpoint_path, model, optimizer):
    """Load training checkpoint"""
    if os.path.exists(checkpoint_path):
        print(f"📂 Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return (checkpoint['epoch'], checkpoint.get('train_losses', []),
                checkpoint.get('val_losses', []), checkpoint.get('train_ious', []),
                checkpoint.get('val_ious', []), checkpoint.get('train_class_accuracies', []),
                checkpoint.get('val_class_accuracies', []))
    return 0, [], [], [], [], [], []


def save_metrics(metrics, metrics_path):
    """Save training metrics to JSON"""
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"📊 Metrics saved: {metrics_path}")


def visualize_predictions(images, targets, outputs, num_images=1):
    """Enhanced visualization with class information"""
    images = [img.cpu().permute(1, 2, 0).numpy() for img in images[:num_images]]
    targets = targets[:num_images]
    outputs = outputs[:num_images]

    class_names = {0: 'background', 1: 'fake', 2: 'real'}
    colors = {1: 'Reds', 2: 'Blues'}

    for i in range(num_images):
        img = images[i]
        target = targets[i]
        output = outputs[i]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Ground truth
        ax1.imshow(img)
        ax1.set_title("Ground Truth")
        gt_masks = target['masks'].cpu().numpy()
        gt_labels = target['labels'].cpu().numpy()

        for m, label in zip(gt_masks, gt_labels):
            if label in colors:
                ax1.imshow(np.ma.masked_where(m == 0, m), cmap=colors[label], alpha=0.4)

        # Predictions
        ax2.imshow(img)
        ax2.set_title("Predictions")
        pred_masks = output['masks'].detach().cpu().numpy()
        pred_labels = output['labels'].detach().cpu().numpy()
        scores = output['scores'].detach().cpu().numpy()

        for m, label, score in zip(pred_masks, pred_labels, scores):
            if score > 0.5 and label in colors:
                mask = m[0]
                ax2.imshow(np.ma.masked_where(mask < 0.5, mask), cmap=colors[label], alpha=0.4)

        ax1.axis('off')
        ax2.axis('off')
        plt.tight_layout()
        plt.show()


def plot_comprehensive_results(train_losses, val_losses, train_ious, val_ious,
                               train_class_acc, val_class_acc):
    """Create comprehensive training plots"""
    epochs = range(1, len(train_losses) + 1)

    # Create a large figure with subplots
    fig = plt.figure(figsize=(20, 15))

    # 1. Loss curves
    plt.subplot(3, 2, 1)
    plt.plot(epochs, train_losses, 'b-o', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-o', label='Validation Loss', linewidth=2)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. IoU curves
    plt.subplot(3, 2, 2)
    plt.plot(epochs, train_ious, 'g-o', label='Training IoU', linewidth=2)
    plt.plot(epochs, val_ious, 'orange', marker='o', label='Validation IoU', linewidth=2)
    plt.title('Training and Validation IoU', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('IoU Score')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 3. Class accuracies - Fake class
    plt.subplot(3, 2, 3)
    train_fake_acc = [acc.get(1, 0) for acc in train_class_acc]
    val_fake_acc = [acc.get(1, 0) for acc in val_class_acc]
    plt.plot(epochs, train_fake_acc, 'purple', marker='o', label='Training Fake Acc', linewidth=2)
    plt.plot(epochs, val_fake_acc, 'pink', marker='o', label='Validation Fake Acc', linewidth=2)
    plt.title('Fake Class Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 4. Class accuracies - Real class
    plt.subplot(3, 2, 4)
    train_real_acc = [acc.get(2, 0) for acc in train_class_acc]
    val_real_acc = [acc.get(2, 0) for acc in val_class_acc]
    plt.plot(epochs, train_real_acc, 'brown', marker='o', label='Training Real Acc', linewidth=2)
    plt.plot(epochs, val_real_acc, 'gold', marker='o', label='Validation Real Acc', linewidth=2)
    plt.title('Real Class Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 5. Combined Loss and IoU
    plt.subplot(3, 2, 5)
    ax1 = plt.gca()
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='b')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.plot(epochs, train_ious, 'g-', label='Train IoU')
    ax2.plot(epochs, val_ious, 'orange', label='Val IoU')
    ax2.set_ylabel('IoU', color='g')
    ax2.legend(loc='upper right')
    plt.title('Loss vs IoU Comparison', fontsize=14, fontweight='bold')

    # 6. Performance summary
    plt.subplot(3, 2, 6)
    final_metrics = [
        val_losses[-1] if val_losses else 0,
        val_ious[-1] if val_ious else 0,
        val_fake_acc[-1] if val_fake_acc else 0,
        val_real_acc[-1] if val_real_acc else 0
    ]
    metric_names = ['Val Loss', 'Val IoU', 'Fake Acc', 'Real Acc']
    colors_bar = ['red', 'orange', 'purple', 'brown']

    bars = plt.bar(metric_names, final_metrics, color=colors_bar, alpha=0.7)
    plt.title('Final Validation Metrics', fontsize=14, fontweight='bold')
    plt.ylabel('Score')

    # Add value labels on bars
    for bar, value in zip(bars, final_metrics):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('comprehensive_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    # Updated parameters for GTX 1650 Ti with higher batch size
    args = argparse.Namespace(
        epochs=15,
        lr=1e-4,  # Extremely low LR
        batch_size=4,  # Single sample to isolate issues
        workers=2,  # No multiprocessing
        use_scheduler=True,  # No scheduler
        weight_decay=5e-4,  # No regularization initially
        checkpoint_freq=2,
        early_stopping_patience=4,  # More patience for slower learning
        train_img_dir="D:/Deepfake/datasets/Train",
        train_ann_file="D:/Deepfake/datasets/Annotations/Train_poly_cleaned.json",
        val_img_dir="D:/Deepfake/datasets/Val",
        val_ann_file="D:/Deepfake/datasets/Annotations/Val_poly_cleaned.json",
        checkpoint_path="D:/Deepfake/datasets/checkpoint.pth",
        metrics_path="D:/Deepfake/datasets/metrics.json"
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

        # Clear GPU cache before starting
        torch.cuda.empty_cache()

        # Monitor initial memory usage
        if hasattr(torch.cuda, 'memory_reserved'):
            print(f"Initial GPU memory reserved: {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")

        # Memory safety warning for GTX 1650 Ti
        print("⚠️  WARNING: GTX 1650 Ti has 4GB VRAM. With batch_size=4:")
        print("   - Monitor GPU memory usage closely")
        print("   - If you get OOM errors, reduce batch_size to 2 or image size to 256x256")
        print("   - Consider using gradient accumulation if training is unstable")

    # Verify paths before creating datasets
    print("\n🔍 Verifying paths...")
    required_paths = [args.train_img_dir, args.train_ann_file, args.val_img_dir, args.val_ann_file]
    for path in required_paths:
        print(f"{path} - Exists: {os.path.exists(path)}")

    # Create datasets - using 320x320 to balance performance and memory
    print("\n📦 Creating datasets...")
    train_ds = CocoDataset(args.train_img_dir, args.train_ann_file, resize_size=(256, 256) ,max_samples=5000)
    val_ds = CocoDataset(args.val_img_dir, args.val_ann_file, resize_size=(256, 256), max_samples=800)

    # Create data loaders
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.workers, pin_memory=True
    )

    print(f"\n📊 Dataset stats:")
    print(f"Training images: {len(train_ds)}")
    print(f"Validation images: {len(val_ds)}")

    # Initialize model for 3 classes (0: background, 1: fake, 2: real)
    model = get_model(num_classes=3).to(device)

    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Learning rate scheduler
    if args.use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )

    # Load checkpoint if exists
    start_epoch, train_losses, val_losses, train_ious, val_ious, train_class_acc, val_class_acc = \
        load_checkpoint(args.checkpoint_path, model, optimizer)

    # Training variables
    best_val_loss = float('inf')
    best_iou = 0.0
    start_time = time.time()
    epochs_without_improvement = 0  # For early stopping

    print(f"\n🏋️ Starting training from epoch {start_epoch + 1}...")

    for epoch in trange(start_epoch + 1, args.epochs + 1, desc='Epochs'):
        epoch_start_time = time.time()


        # Training phase
        model.train()
        epoch_train_loss = 0.0
        num_batches = 0

        for batch_idx, (images, targets) in enumerate(train_loader):
            if len(images) == 0:
                continue

            # Memory monitoring for GTX 1650 Ti
            if torch.cuda.is_available() and batch_idx == 0:
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, 'memory_allocated'):
                    mem_allocated = torch.cuda.memory_allocated(0) / 1024 ** 3
                    if mem_allocated > 3.5:  # 3.5GB threshold for 4GB GPU
                        print(f"⚠️  High GPU memory usage: {mem_allocated:.2f} GB")

            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            try:
                with torch.cuda.amp.autocast():
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())

                optimizer.zero_grad()
                scaler.scale(losses).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                epoch_train_loss += losses.item()
                num_batches += 1

                if (batch_idx + 1) % 10 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"Epoch {epoch}, Batch {batch_idx + 1}, Loss: {losses.item():.4f}, LR: {current_lr:.2e}")

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"🚨 GPU Out of Memory at batch {batch_idx + 1}!")
                    print("💡 Try reducing batch_size to 2 or image_size to 256x256")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    raise e
                else:
                    raise e

            # Clear cache every 20 batches to prevent memory buildup
            if torch.cuda.is_available() and (batch_idx + 1) % 20 == 0:
                torch.cuda.empty_cache()

        # Calculate training metrics
        avg_train_loss = epoch_train_loss / num_batches if num_batches > 0 else float('inf')
        train_iou, train_class_accuracy = evaluate_model(model, train_loader, device)

        train_losses.append(avg_train_loss)
        train_ious.append(train_iou)
        train_class_acc.append(train_class_accuracy)

        # Validation phase
        model.eval()
        val_loss = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for val_images, val_targets in val_loader:
                if len(val_images) == 0:
                    continue

                val_images = [img.to(device) for img in val_images]
                val_targets = [{k: v.to(device) for k, v in t.items()} for t in val_targets]

                # Calculate validation loss
                model.train()
                loss_dict = model(val_images, val_targets)
                model.eval()

                if isinstance(loss_dict, dict):
                    val_loss += sum(loss for loss in loss_dict.values()).item()
                    num_val_batches += 1

        avg_val_loss = val_loss / num_val_batches if num_val_batches > 0 else float('inf')
        val_iou, val_class_accuracy = evaluate_model(model, val_loader, device)

        val_losses.append(avg_val_loss)
        val_ious.append(val_iou)
        val_class_acc.append(val_class_accuracy)

        # Update learning rate
        if args.use_scheduler:
            scheduler.step(avg_val_loss)

        # Print epoch results
        epoch_time = time.time() - epoch_start_time
        print(f"\n📈 Epoch {epoch} Results ({epoch_time:.1f}s):")
        print(f"  🟢 Train Loss: {avg_train_loss:.4f}, Train IoU: {train_iou:.4f}")
        print(f"  🔍 Val Loss: {avg_val_loss:.4f}, Val IoU: {val_iou:.4f}")
        print(f"  🎯 Train Class Acc: {train_class_accuracy}")
        print(f"  🎯 Val Class Acc: {val_class_accuracy}")

        # Save best models
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_maskrcnn_loss800.pth")
            print(f"✅ Saved best loss model: {best_val_loss:.4f}")

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), "best_maskrcnn_iou800.pth")
            print(f"✅ Saved best IoU model: {best_iou:.4f}")

        # Save checkpoint
        if epoch % args.checkpoint_freq == 0 or epoch == args.epochs:
            save_checkpoint(model, optimizer, epoch, train_losses, val_losses,
                            train_ious, val_ious, train_class_acc, val_class_acc,
                            args.checkpoint_path)

        # Memory cleanup and monitoring
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated(0) / 1024 ** 3
            if memory_used > 3.5:
                print(f"⚠️  High memory usage after epoch {epoch}: {memory_used:.2f} GB")
            torch.cuda.empty_cache()

    # Final save
    torch.save(model.state_dict(), "final_maskrcnn800.pth")

    # Calculate final metrics
    total_time = time.time() - start_time
    final_metrics = {
        "training_completed": datetime.now().isoformat(),
        "total_training_time_hours": total_time / 3600,
        "total_epochs": args.epochs,
        "best_validation_loss": best_val_loss,
        "best_validation_iou": best_iou,
        "final_train_loss": train_losses[-1] if train_losses else None,
        "final_val_loss": val_losses[-1] if val_losses else None,
        "final_train_iou": train_ious[-1] if train_ious else None,
        "final_val_iou": val_ious[-1] if val_ious else None,
        "final_train_class_accuracies": train_class_acc[-1] if train_class_acc else None,
        "final_val_class_accuracies": val_class_acc[-1] if val_class_acc else None,
        "training_history": {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_ious": train_ious,
            "val_ious": val_ious,
            "train_class_accuracies": train_class_acc,
            "val_class_accuracies": val_class_acc
        },
        "model_config": {
            "num_classes": 3,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "image_size": [320, 320]
        }
    }

    # Save metrics
    save_metrics(final_metrics, args.metrics_path)

    print(f"\n🎉 Training complete! Total time: {total_time / 3600:.2f} hours")
    print(f"📊 Best validation loss: {best_val_loss:.4f}")
    print(f"📊 Best validation IoU: {best_iou:.4f}")

    # Create comprehensive plots
    if train_losses and val_losses:
        plot_comprehensive_results(train_losses, val_losses, train_ious, val_ious,
                                   train_class_acc, val_class_acc)

    print("✅ All results saved successfully!")