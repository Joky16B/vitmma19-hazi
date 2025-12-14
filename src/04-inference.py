"""
Inference script for ankle alignment classification.
Runs predictions on test images and displays results.
"""
import torch
import numpy as np
from pathlib import Path
from torchvision import datasets, transforms
import argparse

from config import IMAGE_SIZE, NUM_CLASSES, PROCESSED_DATA_DIR, MODEL_SAVE_PATH, CLASS_NAMES
from models import create_model
from utils import setup_logger

logger = setup_logger(__name__)


def load_model(model_path: str, device: torch.device) -> torch.nn.Module:
    """
    Load a trained model from checkpoint.
    
    Args:
        model_path: Path to saved model checkpoint
        device: Device to load model on
    
    Returns:
        Loaded model in eval mode
    """
    logger.info(f"Loading model from: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    model = create_model('anklenet', num_classes=NUM_CLASSES)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    logger.info(f"Model loaded successfully")
    logger.info(f"  Trained for {checkpoint.get('epoch', 'unknown')} epochs")
    
    val_acc = checkpoint.get('val_acc')
    if val_acc is not None and isinstance(val_acc, (int, float)):
        logger.info(f"  Validation accuracy: {val_acc:.2f}%")
    else:
        logger.info(f"  Validation accuracy: Not available")
    
    return model


def get_test_dataset(processed_dir: Path):
    """Load test dataset from ImageFolder structure."""
    test_dir = processed_dir / 'test'
    
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.ImageFolder(test_dir, transform=transform)
    logger.info(f"Loaded {len(dataset)} test images")
    
    return dataset


def predict_samples(model: torch.nn.Module, 
                   dataset: datasets.ImageFolder,
                   device: torch.device,
                   n_samples: int = 10):
    """
    Run predictions on random samples from the test set.
    
    Args:
        model: Trained model
        dataset: Test dataset
        device: Device for inference
        n_samples: Number of samples to predict
    """
    n_samples = min(n_samples, len(dataset))
    
    indices = np.random.choice(len(dataset), n_samples, replace=False)
    
    logger.info(f"\nRunning inference on {n_samples} random test images...")
    logger.info("=" * 80)
    logger.info(f"{'#':<4} {'Predicted':<15} {'True Label':<15} {'Confidence':<12} {'Status':<8}")
    logger.info("=" * 80)
    
    correct = 0
    
    with torch.no_grad():
        for idx, sample_idx in enumerate(indices, 1):
            image, true_label_idx = dataset[sample_idx]
            true_label = CLASS_NAMES[true_label_idx]
            
            image_path = Path(dataset.imgs[sample_idx][0]).name
            
            image_batch = image.unsqueeze(0).to(device)
            
            outputs = model(image_batch)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            confidence, predicted_idx = torch.max(probabilities, 1)
            predicted_idx = predicted_idx.item()
            confidence = confidence.item() * 100
            
            predicted_label = CLASS_NAMES[predicted_idx]
            status = "✓ PASS" if predicted_idx == true_label_idx else "✗ FAIL"
            
            if predicted_idx == true_label_idx:
                correct += 1
            
            logger.info(
                f"{idx:<4} {predicted_label:<15} {true_label:<15} "
                f"{confidence:>6.2f}%     {status:<8}"
            )
    
    logger.info("=" * 80)
    accuracy = 100 * correct / n_samples
    logger.info(f"Sample Accuracy: {correct}/{n_samples} ({accuracy:.2f}%)")
    logger.info("=" * 80)


def predict_single_image(model: torch.nn.Module,
                        image_path: str,
                        device: torch.device):
    """
    Run prediction on a single image file.
    
    Args:
        model: Trained model
        image_path: Path to image file
        device: Device for inference
    """
    from PIL import Image
    
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
    
    logger.info(f"\nPrediction for: {Path(image_path).name}")
    logger.info("=" * 50)
    
    for idx, class_name in enumerate(CLASS_NAMES):
        prob = probabilities[idx].item() * 100
        bar = "█" * int(prob / 2)
        logger.info(f"  {class_name:<12} {prob:>6.2f}% {bar}")
    
    predicted_idx = torch.argmax(probabilities).item()
    logger.info("=" * 50)
    logger.info(f"Predicted class: {CLASS_NAMES[predicted_idx]}")


def main():
    parser = argparse.ArgumentParser(description='Run inference on ankle alignment images')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to trained model checkpoint')
    parser.add_argument('--image', type=str, default=None,
                       help='Path to single image for prediction')
    parser.add_argument('--n-samples', type=int, default=10,
                       help='Number of random test samples to predict')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Path to processed data directory')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    model_path = args.model_path if args.model_path else MODEL_SAVE_PATH
    
    if not Path(model_path).exists():
        logger.error(f"Model not found at: {model_path}")
        logger.info("Please train a model first using 02-training.py")
        return
    
    model = load_model(model_path, device)
    
    if args.image:
        predict_single_image(model, args.image, device)
        return
    
    data_dir = Path(args.data_dir) if args.data_dir else Path(PROCESSED_DATA_DIR)
    
    try:
        test_dataset = get_test_dataset(data_dir)
        predict_samples(model, test_dataset, device, n_samples=args.n_samples)
    except FileNotFoundError as e:
        logger.error(str(e))
        logger.info("Please run 01-data-preprocessing.py first to create the dataset")


if __name__ == '__main__':
    main()
