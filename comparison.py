import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
from pycocotools.coco import COCO
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from PIL import Image
import pandas as pd
from datetime import datetime
import time

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ModelComparisonTool:
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.metrics = {}
        self.model_configs = {
            'subset_1000': {
                'name': 'Subset 1K Training',
                'train_samples': 1000,
                'val_samples': 100,
                'color': '#FF6B6B',
                'marker': 'o'
            },
            'subset_2500': {
                'name': 'Subset 2.5K Training',
                'train_samples': 2500,
                'val_samples': 300,
                'color': '#4ECDC4',
                'marker': 's'
            },
            'subset_5000': {
                'name': 'Subset 5K Training',
                'train_samples': 5000,
                'val_samples': 800,
                'color': '#45B7D1',
                'marker': '^'
            }
        }

    def get_model(self, num_classes=3):
        """Create MaskRCNN model architecture"""
        model = maskrcnn_resnet50_fpn(weights="DEFAULT")
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
        return model

    def load_model_and_metrics(self, model_path, metrics_path, model_key):
        """Load trained model and its metrics"""
        print(f"Loading {model_key} model...")

        # Load model
        model = self.get_model().to(self.device)
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint)
            model.eval()
            self.models[model_key] = model
            print(f"✅ Model loaded: {model_path}")
        else:
            print(f"❌ Model not found: {model_path}")
            return False

        # Load metrics
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                self.metrics[model_key] = json.load(f)
            print(f"✅ Metrics loaded: {metrics_path}")
        else:
            print(f"❌ Metrics not found: {metrics_path}")
            return False

        return True

    def calculate_iou(self, pred_masks, target_masks, threshold=0.5):
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

    def evaluate_model_comprehensive(self, model, test_loader, model_name):
        """Comprehensive evaluation of a single model"""
        model.eval()

        metrics = {
            'total_samples': 0,
            'total_iou': 0.0,
            'class_correct': {1: 0, 2: 0},  # fake, real
            'class_total': {1: 0, 2: 0},
            'confidence_scores': [],
            'processing_times': [],
            'detection_counts': []
        }

        print(f"🔍 Evaluating {model_name}...")

        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(test_loader):
                if len(images) == 0:
                    continue

                batch_start = time.time()
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                outputs = model(images)
                batch_time = time.time() - batch_start
                metrics['processing_times'].append(batch_time)

                for output, target in zip(outputs, targets):
                    # IoU calculation
                    if len(output['masks']) > 0 and len(target['masks']) > 0:
                        iou = self.calculate_iou(
                            output['masks'][:, 0],
                            target['masks'],
                            threshold=0.5
                        )
                        metrics['total_iou'] += iou
                        metrics['total_samples'] += 1

                    # Class accuracy
                    high_conf_mask = output['scores'] > 0.5
                    pred_labels = output['labels'][high_conf_mask]
                    pred_scores = output['scores'][high_conf_mask]

                    metrics['detection_counts'].append(len(pred_labels))
                    if len(pred_scores) > 0:
                        metrics['confidence_scores'].extend(pred_scores.cpu().numpy())

                    target_labels = target['labels']
                    for label in target_labels:
                        label_val = label.item()
                        if label_val in metrics['class_total']:
                            metrics['class_total'][label_val] += 1
                            if label_val in pred_labels:
                                metrics['class_correct'][label_val] += 1

                if (batch_idx + 1) % 10 == 0:
                    print(f"  Processed {batch_idx + 1} batches...")

        # Calculate final metrics
        avg_iou = metrics['total_iou'] / metrics['total_samples'] if metrics['total_samples'] > 0 else 0.0
        class_accuracies = {
            k: metrics['class_correct'][k] / metrics['class_total'][k] if metrics['class_total'][k] > 0 else 0.0
            for k in metrics['class_total']
        }

        result = {
            'avg_iou': avg_iou,
            'class_accuracies': class_accuracies,
            'avg_confidence': np.mean(metrics['confidence_scores']) if metrics['confidence_scores'] else 0.0,
            'avg_processing_time': np.mean(metrics['processing_times']),
            'avg_detections_per_image': np.mean(metrics['detection_counts']),
            'total_samples_processed': metrics['total_samples']
        }

        return result

    def create_comparison_plots(self):
        """Create comprehensive comparison plots"""
        if not self.metrics:
            print("❌ No metrics loaded for comparison!")
            return

        # Create large figure with multiple subplots
        fig = plt.figure(figsize=(24, 18))

        # 1. Training Loss Comparison
        plt.subplot(3, 4, 1)
        for key, config in self.model_configs.items():
            if key in self.metrics:
                history = self.metrics[key].get('training_history', {})
                train_losses = history.get('train_losses', [])
                if train_losses:
                    epochs = range(1, len(train_losses) + 1)
                    plt.plot(epochs, train_losses, color=config['color'],
                             marker=config['marker'], label=config['name'], linewidth=2)

        plt.title('Training Loss Comparison', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Training Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 2. Validation Loss Comparison
        plt.subplot(3, 4, 2)
        for key, config in self.model_configs.items():
            if key in self.metrics:
                history = self.metrics[key].get('training_history', {})
                val_losses = history.get('val_losses', [])
                if val_losses:
                    epochs = range(1, len(val_losses) + 1)
                    plt.plot(epochs, val_losses, color=config['color'],
                             marker=config['marker'], label=config['name'], linewidth=2)

        plt.title('Validation Loss Comparison', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 3. IoU Comparison
        plt.subplot(3, 4, 3)
        for key, config in self.model_configs.items():
            if key in self.metrics:
                history = self.metrics[key].get('training_history', {})
                val_ious = history.get('val_ious', [])
                if val_ious:
                    epochs = range(1, len(val_ious) + 1)
                    plt.plot(epochs, val_ious, color=config['color'],
                             marker=config['marker'], label=config['name'], linewidth=2)

        plt.title('Validation IoU Comparison', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('IoU Score')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 4. Final Performance Bar Chart
        plt.subplot(3, 4, 4)
        models = []
        final_ious = []
        final_losses = []

        for key, config in self.model_configs.items():
            if key in self.metrics:
                models.append(f"{config['train_samples']}+{config['val_samples']}")
                final_ious.append(self.metrics[key].get('final_val_iou', 0))
                final_losses.append(self.metrics[key].get('final_val_loss', 0))

        x = np.arange(len(models))
        width = 0.35

        plt.bar(x - width / 2, final_ious, width, label='Final IoU', alpha=0.8)
        plt.bar(x + width / 2, [l / 10 for l in final_losses], width, label='Final Loss (/10)', alpha=0.8)

        plt.title('Final Performance Comparison', fontsize=14, fontweight='bold')
        plt.xlabel('Dataset Size (Train+Val)')
        plt.ylabel('Score')
        plt.xticks(x, models)
        plt.legend()

        # 5. Convergence Speed Analysis
        plt.subplot(3, 4, 5)
        convergence_epochs = []
        dataset_sizes = []

        for key, config in self.model_configs.items():
            if key in self.metrics:
                history = self.metrics[key].get('training_history', {})
                val_losses = history.get('val_losses', [])
                if val_losses:
                    # Find epoch where loss stabilizes (within 5% of minimum)
                    min_loss = min(val_losses)
                    threshold = min_loss * 1.05
                    for i, loss in enumerate(val_losses):
                        if loss <= threshold:
                            convergence_epochs.append(i + 1)
                            dataset_sizes.append(config['train_samples'])
                            break

        if convergence_epochs:
            plt.scatter(dataset_sizes, convergence_epochs, s=100, alpha=0.7)
            plt.plot(dataset_sizes, convergence_epochs, '--', alpha=0.5)
            plt.title('Convergence Speed vs Dataset Size', fontsize=14, fontweight='bold')
            plt.xlabel('Training Dataset Size')
            plt.ylabel('Epochs to Convergence')
            plt.grid(True, alpha=0.3)

        # 6. Class Accuracy Comparison - Fake Detection
        plt.subplot(3, 4, 6)
        for key, config in self.model_configs.items():
            if key in self.metrics:
                history = self.metrics[key].get('training_history', {})
                val_class_acc = history.get('val_class_accuracies', [])
                if val_class_acc:
                    fake_acc = [acc.get(1, 0) for acc in val_class_acc]
                    epochs = range(1, len(fake_acc) + 1)
                    plt.plot(epochs, fake_acc, color=config['color'],
                             marker=config['marker'], label=config['name'], linewidth=2)

        plt.title('Fake Detection Accuracy', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 7. Class Accuracy Comparison - Real Detection
        plt.subplot(3, 4, 7)
        for key, config in self.model_configs.items():
            if key in self.metrics:
                history = self.metrics[key].get('training_history', {})
                val_class_acc = history.get('val_class_accuracies', [])
                if val_class_acc:
                    real_acc = [acc.get(2, 0) for acc in val_class_acc]
                    epochs = range(1, len(real_acc) + 1)
                    plt.plot(epochs, real_acc, color=config['color'],
                             marker=config['marker'], label=config['name'], linewidth=2)

        plt.title('Real Detection Accuracy', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 8. Training Time Analysis
        plt.subplot(3, 4, 8)
        training_times = []
        dataset_sizes = []

        for key, config in self.model_configs.items():
            if key in self.metrics:
                train_time = self.metrics[key].get('total_training_time_hours', 0)
                if train_time > 0:
                    training_times.append(train_time)
                    dataset_sizes.append(config['train_samples'])

        if training_times:
            plt.scatter(dataset_sizes, training_times, s=100, alpha=0.7)
            plt.plot(dataset_sizes, training_times, '--', alpha=0.5)
            plt.title('Training Time vs Dataset Size', fontsize=14, fontweight='bold')
            plt.xlabel('Training Dataset Size')
            plt.ylabel('Training Time (Hours)')
            plt.grid(True, alpha=0.3)

        # 9. Overfitting Analysis (Train vs Val Loss)
        plt.subplot(3, 4, 9)
        for key, config in self.model_configs.items():
            if key in self.metrics:
                history = self.metrics[key].get('training_history', {})
                train_losses = history.get('train_losses', [])
                val_losses = history.get('val_losses', [])

                if train_losses and val_losses:
                    min_len = min(len(train_losses), len(val_losses))
                    gap = [val_losses[i] - train_losses[i] for i in range(min_len)]
                    epochs = range(1, min_len + 1)
                    plt.plot(epochs, gap, color=config['color'],
                             marker=config['marker'], label=config['name'], linewidth=2)

        plt.title('Overfitting Analysis (Val - Train Loss)', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Gap')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)

        # 10. Best Performance Summary
        plt.subplot(3, 4, 10)
        best_metrics = {
            'IoU': [],
            'Loss': [],
            'Fake Acc': [],
            'Real Acc': []
        }

        model_names = []
        for key, config in self.model_configs.items():
            if key in self.metrics:
                model_names.append(f"{config['train_samples']}")
                best_metrics['IoU'].append(self.metrics[key].get('best_validation_iou', 0))
                best_metrics['Loss'].append(self.metrics[key].get('best_validation_loss', 0))

                final_val_acc = self.metrics[key].get('final_val_class_accuracies', {})
                best_metrics['Fake Acc'].append(final_val_acc.get(1, 0))
                best_metrics['Real Acc'].append(final_val_acc.get(2, 0))

        x = np.arange(len(model_names))
        width = 0.2

        plt.bar(x - 1.5 * width, best_metrics['IoU'], width, label='Best IoU', alpha=0.8)
        plt.bar(x - 0.5 * width, [l / 10 for l in best_metrics['Loss']], width, label='Best Loss (/10)', alpha=0.8)
        plt.bar(x + 0.5 * width, best_metrics['Fake Acc'], width, label='Fake Acc', alpha=0.8)
        plt.bar(x + 1.5 * width, best_metrics['Real Acc'], width, label='Real Acc', alpha=0.8)

        plt.title('Best Performance Summary', fontsize=14, fontweight='bold')
        plt.xlabel('Training Dataset Size')
        plt.ylabel('Score')
        plt.xticks(x, model_names)
        plt.legend()

        # 11. Learning Rate Effect (if available)
        plt.subplot(3, 4, 11)
        plt.text(0.5, 0.5, 'Learning Rate\nAnalysis\n(Custom Analysis)',
                 ha='center', va='center', transform=plt.gca().transAxes,
                 fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        plt.title('Learning Rate Analysis', fontsize=14, fontweight='bold')
        plt.axis('off')

        # 12. Memory Usage Analysis
        plt.subplot(3, 4, 12)
        plt.text(0.5, 0.5, 'Memory Usage\nAnalysis\n(Add GPU Memory\nTracking)',
                 ha='center', va='center', transform=plt.gca().transAxes,
                 fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
        plt.title('Memory Usage Analysis', fontsize=14, fontweight='bold')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig('comprehensive_subset_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

    def generate_comparison_report(self):
        """Generate detailed comparison report"""
        if not self.metrics:
            print("❌ No metrics available for report generation!")
            return

        print("\n" + "=" * 80)
        print("🔍 COMPREHENSIVE SUBSET TRAINING COMPARISON REPORT")
        print("=" * 80)

        # Create comparison table
        comparison_data = []

        for key, config in self.model_configs.items():
            if key in self.metrics:
                metrics = self.metrics[key]

                row = {
                    'Model': config['name'],
                    'Train Size': config['train_samples'],
                    'Val Size': config['val_samples'],
                    'Best Val Loss': metrics.get('best_validation_loss', 'N/A'),
                    'Best Val IoU': metrics.get('best_validation_iou', 'N/A'),
                    'Final Train Loss': metrics.get('final_train_loss', 'N/A'),
                    'Final Val Loss': metrics.get('final_val_loss', 'N/A'),
                    'Final Val IoU': metrics.get('final_val_iou', 'N/A'),
                    'Training Time (hrs)': metrics.get('total_training_time_hours', 'N/A'),
                    'Total Epochs': metrics.get('total_epochs', 'N/A')
                }

                # Add class accuracies
                final_class_acc = metrics.get('final_val_class_accuracies', {})
                row['Fake Detection Acc'] = final_class_acc.get(1, 'N/A')
                row['Real Detection Acc'] = final_class_acc.get(2, 'N/A')

                comparison_data.append(row)

        # Create DataFrame and display
        df = pd.DataFrame(comparison_data)

        print("\n📊 PERFORMANCE COMPARISON TABLE:")
        print("-" * 80)
        print(df.to_string(index=False, float_format='%.4f'))

        # Analysis insights
        print("\n🔍 KEY INSIGHTS:")
        print("-" * 40)

        if len(comparison_data) > 1:
            # Best performing model
            best_iou_idx = df['Best Val IoU'].astype(float).idxmax()
            best_loss_idx = df['Best Val Loss'].astype(float).idxmin()

            print(
                f"🏆 Best IoU Performance: {df.iloc[best_iou_idx]['Model']} (IoU: {df.iloc[best_iou_idx]['Best Val IoU']:.4f})")
            print(
                f"🏆 Best Loss Performance: {df.iloc[best_loss_idx]['Model']} (Loss: {df.iloc[best_loss_idx]['Best Val Loss']:.4f})")

            # Data scaling insights
            train_sizes = df['Train Size'].values
            val_ious = df['Best Val IoU'].astype(float).values

            if len(train_sizes) >= 2:
                iou_improvement = val_ious[-1] - val_ious[0]
                size_ratio = train_sizes[-1] / train_sizes[0]
                print(f"📈 IoU Improvement: {iou_improvement:.4f} with {size_ratio:.1f}x more training data")

            # Efficiency analysis
            training_times = df['Training Time (hrs)'].astype(float).values
            if len(training_times) >= 2:
                time_ratio = training_times[-1] / training_times[0]
                print(f"⏱️ Training Time Scaling: {time_ratio:.1f}x longer for largest dataset")

        # Save report
        report_path = f"subset_comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump({
                'comparison_data': comparison_data,
                'generation_time': datetime.now().isoformat(),
                'models_compared': list(self.model_configs.keys())
            }, f, indent=4)

        print(f"\n💾 Report saved: {report_path}")

        return df


def main():
    """Main execution function"""
    print("🔍 Subset Training Comparison Tool")
    print("=" * 50)

    # Initialize comparison tool
    comparator = ModelComparisonTool()

    # Define paths for your models and metrics
    model_paths = {
        'subset_1000': {
            'model': 'C:/Users/bogdan/PycharmProjects/Licence/best_maskrcnn_iou100.pth',  # Update with actual path
            'metrics': 'D:/Deepfake/datasets/Annotations/metrics1/metrics2.json'
        },
        'subset_2500': {
            'model': 'C:/Users/bogdan/PycharmProjects/Licence/best_maskrcnn_iou300.pth',  # Update with actual path
            'metrics': 'D:/Deepfake/datasets/Annotations/metrics1/metrics1.json'
        },
        'subset_5000': {
            'model': 'C:/Users/bogdan/PycharmProjects/Licence/best_maskrcnn_iou800.pth',  # Update with actual path
            'metrics': 'D:/Deepfake/datasets/Annotations/metrics1/metrics3.json'
        }
    }

    # Load all models and metrics
    loaded_models = 0
    for key, paths in model_paths.items():
        if comparator.load_model_and_metrics(paths['model'], paths['metrics'], key):
            loaded_models += 1

    print(f"\n✅ Successfully loaded {loaded_models} models for comparison")

    if loaded_models == 0:
        print("❌ No models loaded! Please check file paths.")
        return

    # Generate comparison plots
    print("\n📊 Generating comparison plots...")
    comparator.create_comparison_plots()

    # Generate detailed report
    print("\n📋 Generating comparison report...")
    comparison_df = comparator.generate_comparison_report()

    print("\n✅ Comparison analysis complete!")
    print("\nFiles generated:")
    print("- comprehensive_subset_comparison.png")
    print("- subset_comparison_report_[timestamp].json")


if __name__ == "__main__":
    main()