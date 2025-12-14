# Data preprocessing script
# This script handles data loading, cleaning, and transformation.
# Parses Label Studio JSON exports and prepares data for training.

import json
import os
import re
import shutil
from pathlib import Path
from collections import defaultdict

import pandas as pd
from PIL import Image, UnidentifiedImageError

from utils import setup_logger
from config import DATA_DIR, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED

logger = setup_logger(__name__)

CLASS_MAPPING = {
    "1_Pronacio": 0,
    "2_Neutralis": 1, 
    "3_Szupinacio": 2,
}
CLASS_NAMES = ["Pronation", "Neutral", "Supination"]


def extract_original_filename(file_upload: str) -> str:
    """
    Extract original filename from Label Studio's file_upload field.
    Label Studio prefixes filenames with a UUID, e.g., 'd1a7dc20-internet_actualne_01.jpg'
    """
    # Pattern: UUID-originalfilename.ext
    match = re.match(r'^[a-f0-9]{8}-(.+)$', file_upload)
    if match:
        return match.group(1)
    return file_upload


def parse_label_studio_json(json_path: Path) -> list[dict]:
    """
    Parse a Label Studio JSON export file and extract image-label pairs.
    
    Returns:
        List of dicts with keys: 'original_filename', 'label', 'label_id'
    """
    results = []
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON {json_path}: {e}")
        return results
    
    for task in data:
        file_upload = task.get('file_upload', '')
        annotations = task.get('annotations', [])
        
        if not annotations:
            logger.debug(f"No annotations for task in {json_path}")
            continue
        
        annotation = annotations[0]
        result_list = annotation.get('result', [])
        
        if not result_list:
            continue
            
        for result in result_list:
            if result.get('type') == 'choices':
                choices = result.get('value', {}).get('choices', [])
                if choices:
                    label = choices[0]
                    original_filename = extract_original_filename(file_upload)
                    
                    if label in CLASS_MAPPING:
                        results.append({
                            'original_filename': original_filename,
                            'label': label,
                            'label_id': CLASS_MAPPING[label],
                        })
                    else:
                        logger.warning(f"Unknown label '{label}' in {json_path}")
    
    return results


def find_image_file(folder_path: Path, filename: str) -> Path | None:
    """
    Find an image file in a folder, handling case-insensitivity and extensions.
    """
    direct_path = folder_path / filename
    if direct_path.exists():
        return direct_path
    
    filename_lower = filename.lower()
    for f in folder_path.iterdir():
        if f.is_file() and f.name.lower() == filename_lower:
            return f
    
    base_name = Path(filename).stem
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        test_path = folder_path / (base_name + ext)
        if test_path.exists():
            return test_path
    
    return None


def verify_image(image_path: Path) -> bool:
    """
    Verify that the image can be opened by PIL.
    """
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except (UnidentifiedImageError, OSError, Exception) as e:
        logger.warning(f"Invalid image file {image_path}: {e}")
        return False


def collect_all_data(raw_data_dir: Path) -> pd.DataFrame:
    """
    Collect all labeled data from all student folders.
    
    Returns:
        DataFrame with columns: image_path, label, label_id, source_folder
    """
    all_records = []
    
    for folder in raw_data_dir.iterdir():
        if not folder.is_dir():
            continue
        
        folder_name = folder.name
        
        if folder_name in ['consensus', 'sample']:
            continue
        
        json_files = list(folder.glob('*.json'))
        
        if not json_files:
            logger.debug(f"No JSON files in {folder_name}")
            continue
        
        logger.info(f"Processing folder: {folder_name}")
        
        for json_file in json_files:
            annotations = parse_label_studio_json(json_file)
            
            for ann in annotations:
                image_file = find_image_file(folder, ann['original_filename'])
                
                if image_file:
                    if verify_image(image_file):
                        all_records.append({
                            'image_path': str(image_file),
                            'original_filename': ann['original_filename'],
                            'label': ann['label'],
                            'label_id': ann['label_id'],
                            'source_folder': folder_name,
                        })
                    else:
                        logger.warning(f"Skipping corrupt image: {image_file}")
                else:
                    logger.warning(
                        f"Image not found: {ann['original_filename']} in {folder_name}"
                    )
    
    df = pd.DataFrame(all_records)
    logger.info(f"Collected {len(df)} labeled images from {df['source_folder'].nunique()} folders")
    
    return df


def create_dataset_splits(
    df: pd.DataFrame,
    output_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """
    Split the dataset into train/val/test sets with stratification.
    Copies images into organized folder structure.
    
    Output structure:
        output_dir/
            train/
                Pronation/
                Neutral/
                Supination/
            val/
                ...
            test/
                ...
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    splits = {}
    split_dfs = []
    
    for label_id in df['label_id'].unique():
        label_df = df[df['label_id'] == label_id].copy()
        n = len(label_df)
        
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train_df = label_df.iloc[:n_train].copy()
        val_df = label_df.iloc[n_train:n_train + n_val].copy()
        test_df = label_df.iloc[n_train + n_val:].copy()
        
        train_df['split'] = 'train'
        val_df['split'] = 'val'
        test_df['split'] = 'test'
        
        split_dfs.extend([train_df, val_df, test_df])
    
    full_df = pd.concat(split_dfs, ignore_index=True)
    
    for split_name in ['train', 'val', 'test']:
        split_df = full_df[full_df['split'] == split_name].copy()
        splits[split_name] = split_df
        
        for class_name in CLASS_NAMES:
            (output_dir / split_name / class_name).mkdir(parents=True, exist_ok=True)
    
    new_paths = []
    for idx, row in full_df.iterrows():
        src_path = Path(row['image_path'])
        class_name = CLASS_NAMES[row['label_id']]
        split_name = row['split']
        
        unique_name = f"{row['source_folder']}_{src_path.name}"
        dst_path = output_dir / split_name / class_name / unique_name
        
        if src_path.exists():
            shutil.copy2(src_path, dst_path)
            new_paths.append(str(dst_path))
        else:
            logger.warning(f"Source image not found: {src_path}")
            new_paths.append(None)
    
    full_df['processed_path'] = new_paths
    
    return splits, full_df


def print_dataset_statistics(df: pd.DataFrame):
    """Print summary statistics of the dataset."""
    logger.info("=" * 50)
    logger.info("DATASET STATISTICS")
    logger.info("=" * 50)
    
    logger.info(f"Total images: {len(df)}")
    logger.info(f"Number of source folders: {df['source_folder'].nunique()}")
    
    logger.info("\nClass distribution:")
    for label_id, class_name in enumerate(CLASS_NAMES):
        count = len(df[df['label_id'] == label_id])
        pct = 100 * count / len(df) if len(df) > 0 else 0
        logger.info(f"  {class_name}: {count} ({pct:.1f}%)")
    
    if 'split' in df.columns:
        logger.info("\nSplit distribution:")
        for split_name in ['train', 'val', 'test']:
            split_df = df[df['split'] == split_name]
            logger.info(f"  {split_name}: {len(split_df)} images")
            for label_id, class_name in enumerate(CLASS_NAMES):
                count = len(split_df[split_df['label_id'] == label_id])
                logger.info(f"    {class_name}: {count}")
    
    logger.info("=" * 50)


def preprocess():
    """Main preprocessing pipeline."""
    logger.info("Starting data preprocessing...")
    
    data_dir = Path(DATA_DIR)
    raw_data_dir = data_dir / "anklealign"
    processed_dir = data_dir / "processed"
    
    if not raw_data_dir.exists():
        logger.error(f"Raw data directory not found: {raw_data_dir}")
        logger.info("Please ensure data is mounted at /data/anklealign (Docker) or data/anklealign (local)")
        return
    
    logger.info(f"Raw data directory: {raw_data_dir}")
    logger.info(f"Output directory: {processed_dir}")
    logger.info(f"Split ratios - Train: {TRAIN_RATIO}, Val: {VAL_RATIO}, Test: {TEST_RATIO}")
    logger.info(f"Random seed: {RANDOM_SEED}")
    
    if processed_dir.exists():
        logger.info("Cleaning existing processed directory...")
        shutil.rmtree(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("\n[Step 1/3] Collecting labeled data from all folders...")
    df = collect_all_data(raw_data_dir)
    
    if df.empty:
        logger.error("No labeled data found!")
        return
    
    logger.info("\n[Step 2/3] Creating train/val/test splits...")
    splits, full_df = create_dataset_splits(
        df,
        processed_dir,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        random_seed=RANDOM_SEED,
    )
    
    logger.info("\n[Step 3/3] Saving metadata...")
    
    csv_path = processed_dir / "labels.csv"
    full_df.to_csv(csv_path, index=False)
    logger.info(f"Saved labels to: {csv_path}")
    
    class_info = {
        'class_names': CLASS_NAMES,
        'class_mapping': CLASS_MAPPING,
    }
    with open(processed_dir / "class_info.json", 'w') as f:
        json.dump(class_info, f, indent=2)
    
    print_dataset_statistics(full_df)
    
    logger.info("\nData preprocessing completed successfully!")
    logger.info(f"Processed data saved to: {processed_dir}")


if __name__ == "__main__":
    preprocess()
