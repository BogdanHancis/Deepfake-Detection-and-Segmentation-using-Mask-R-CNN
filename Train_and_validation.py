import torch
import torchvision
from torch.utils.data import DataLoader
from engine import train_one_epoch, evaluate
import utils
import transforms as T
from dataset import OpenForensicsDataset
import matplotlib.pyplot as plt
import numpy as np
import os
import torchvision.transforms.functional as F
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import maskrcnn_resnet50_fpn

# Paths
train_dir = "D:/Deepfake/Train"
val_dir = "D:/Deepfake/Val"
test_dir = "D:/Deepfake/Test-Dev"

# Parameters
num_classes = 2  # 1 class (forgery) + background
num_epochs = 10
batch_size = 2
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# Transforms
def get_transform(train):
    transforms = [T.ToTensor()]
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


# Datasets and loaders
train_dataset = OpenForensicsDataset(train_dir, transforms=get_transform(train=True))
val_dataset = OpenForensicsDataset(val_dir, transforms=get_transform(train=False))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                          collate_fn=utils.collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)


# Model setup
def get_model_instance_segmentation(num_classes):
    model = maskrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model


model = get_model_instance_segmentation(num_classes)
model.to(device)

# Optimizer and LR scheduler
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Training loop with fixed validation loss
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")

    # Train
    model.train()
    train_loss = 0.0
    num_batches = 0
    for images, targets in train_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        train_loss += losses.item()
        num_batches += 1

    train_loss /= num_batches
    train_losses.append(train_loss)

    # Validate
    model.eval()
    val_loss = 0.0
    num_val_batches = 0

    all_val_images = []
    all_val_targets = []
    all_val_outputs = []

    with torch.no_grad():
        for val_images, val_targets in val_loader:
            if len(val_images) == 0:
                continue

            val_images = [img.to(device) for img in val_images]
            val_targets = [{k: v.to(device) for k, v in t.items()} for t in val_targets]

            # 🔧 CORRECTED VALIDATION LOSS COMPUTATION
            loss_dict = model(val_images, val_targets)
            loss = sum(loss for loss in loss_dict.values())
            val_loss += loss.item()
            num_val_batches += 1

            # Only visualize a few samples
            if len(all_val_images) < 2:
                outputs = model(val_images)
                all_val_images.extend(val_images)
                all_val_targets.extend(val_targets)
                all_val_outputs.extend(outputs)

    val_loss /= max(num_val_batches, 1)
    val_losses.append(val_loss)

    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    lr_scheduler.step()

# Save model
torch.save(model.state_dict(), "mask_rcnn_openforensics.pth")

# Plot losses
plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig("loss_curve.png")
plt.show()
