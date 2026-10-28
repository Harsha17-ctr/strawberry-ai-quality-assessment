# Strawberry Dataset Directory

This directory stores raw and processed strawberry datasets for training, validation, testing, and evaluation.

## Dataset Structure
The dataset used in this framework originates from strawberry disease and quality benchmarking datasets (featuring 7 pathology classes and 2-tier severity levels):

1. **Angular Leafspot** (`angular_leafspot`): Bacterial foliar disease (*Xanthomonas fragariae*).
2. **Anthracnose Fruit Rot** (`anthracnose_fruit_rot`): Fungal rot (*Colletotrichum acutatum*).
3. **Blossom Blight** (`blossom_blight`): Infection affecting strawberry flower petals and calyx.
4. **Gray Mold** (`gray_mold`): *Botrytis cinerea* causing soft decay on ripe/unripe fruit.
5. **Leaf Spot** (`leaf_spot`): Fungal foliar spotting (*Mycosphaerella fragariae*).
6. **Powdery Mildew Fruit** (`powdery_mildew_fruit`): White mycelial growth on strawberries (*Podosphaera aphanis*).
7. **Powdery Mildew Leaf** (`powdery_mildew_leaf`): Powdery growth and curling on foliage.

## Severity Ratings:
- **Level 1**: Early/Mild symptom development (small spots, minimal tissue necrosis).
- **Level 2**: Advanced/Severe lesions (extensive fruit rot, severe necrosis).

## Files:
- `raw/`: Unzipped dataset images and Labelme JSON annotations.
- `processed/dataset_manifest.json`: Verified index mapping each image to its class label, split (train/val/test), and severity rating.
