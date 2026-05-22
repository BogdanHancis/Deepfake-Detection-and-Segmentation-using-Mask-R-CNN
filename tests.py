import os
from detectron2.data.datasets import register_coco_instances

def register_datasets():
    # Register training dataset
    register_coco_instances(
        "openforensics_train", {},
        r"D:\Deepfake\Train_poly.json",
        r"D:\Deepfake\Train"
    )

    # Register validation dataset
    register_coco_instances(
        "openforensics_val", {},
        r"D:\Deepfake\Val_poly.json",
        r"D:\Deepfake\Val"
    )
    print("✅ Datasets registered successfully!")

if __name__ == "__main__":
    register_datasets()