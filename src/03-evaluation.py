"""
Model evaluation script.
Evaluates trained model on test set with metrics, confusion matrix, and visualizations.
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import argparse
import json

from config import IMAGE_SIZE, NUM_CLASSES, PROCESSED_DATA_DIR, MODEL_SAVE_PATH, BATCH_SIZE, CLASS_NAMES, OUTPUT_DIR
from models import create_model
from utils import setup_logger

logger = setup_logger(__name__)


def load_model(model_path: str, device: torch.device) -> torch.nn.Module:
    """Load trained model from checkpoint."""
    logger.info(f"Loading model from: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    model = create_model('anklenet', num_classes=NUM_CLASSES)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    logger.info(f"Model loaded (trained {checkpoint.get('epoch', '?')} epochs)")
    
    return model, checkpoint


def get_test_loader(processed_dir: Path, batch_size: int = 32):
    """Create test data loader."""
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
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    logger.info(f"Loaded {len(dataset)} test images")
    
    return loader, dataset


def evaluate_model(model: torch.nn.Module, 
                  test_loader: DataLoader,
                  device: torch.device):
    """
    Evaluate model and collect predictions.
    
    Returns:
        Dictionary with metrics and predictions
    """
    logger.info("Running evaluation...")
    
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probs = []
    total_loss = 0.0
    correct = 0
    total = 0
    
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    accuracy = correct / total
    avg_loss = total_loss / len(test_loader)
    
    results = {
        'accuracy': accuracy,
        'avg_loss': avg_loss,
        'predictions': np.array(all_predictions),
        'labels': np.array(all_labels),
        'probabilities': np.array(all_probs)
    }
    
    logger.info(f"Test Accuracy: {accuracy:.4f} ({correct}/{total})")
    logger.info(f"Average Loss: {avg_loss:.4f}")
    
    return results


def generate_confusion_matrix(results: dict, output_dir: Path):
    """Generate and save confusion matrix plots."""
    y_true = results['labels']
    y_pred = results['predictions']
    
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    im1 = axes[0].imshow(cm, interpolation='nearest', cmap='Blues')
    axes[0].set_title('Confusion Matrix (Counts)', fontsize=12, pad=10)
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')
    
    tick_marks = np.arange(len(CLASS_NAMES))
    axes[0].set_xticks(tick_marks)
    axes[0].set_yticks(tick_marks)
    axes[0].set_xticklabels(CLASS_NAMES, rotation=45)
    axes[0].set_yticklabels(CLASS_NAMES)
    
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[0].text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=14)
    
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(cm_normalized, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    axes[1].set_title('Confusion Matrix (Normalized)', fontsize=12, pad=10)
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_ylabel('True Label')
    
    axes[1].set_xticks(tick_marks)
    axes[1].set_yticks(tick_marks)
    axes[1].set_xticklabels(CLASS_NAMES, rotation=45)
    axes[1].set_yticklabels(CLASS_NAMES)
    
    for i in range(cm_normalized.shape[0]):
        for j in range(cm_normalized.shape[1]):
            value = cm_normalized[i, j]
            axes[1].text(j, i, f'{value:.2f}',
                        ha="center", va="center",
                        color="white" if value > 0.5 else "black",
                        fontsize=14)
    
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Confusion matrix saved to {output_dir / 'confusion_matrix.png'}")
    
    return cm, cm_normalized


def generate_classification_report(results: dict):
    """Generate detailed classification report."""
    y_true = results['labels']
    y_pred = results['predictions']
    
    report_dict = classification_report(
        y_true, y_pred, 
        target_names=CLASS_NAMES,
        output_dict=True,
        digits=4,
        zero_division=0
    )
    
    logger.info("\n" + "="*60)
    logger.info("CLASSIFICATION REPORT")
    logger.info("="*60)
    
    for class_name in CLASS_NAMES:
        metrics = report_dict[class_name]
        logger.info(f"\n{class_name}:")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1-Score:  {metrics['f1-score']:.4f}")
        logger.info(f"  Support:   {int(metrics['support'])}")
    
    logger.info(f"\n{'Overall':}")
    logger.info(f"  Accuracy:  {report_dict['accuracy']:.4f}")
    logger.info(f"  Macro Avg F1: {report_dict['macro avg']['f1-score']:.4f}")
    logger.info("="*60 + "\n")
    
    return report_dict


def plot_per_class_metrics(report_dict: dict, output_dir: Path):
    """Visualize per-class precision, recall, and F1-score."""
    metrics = ['precision', 'recall', 'f1-score']
    class_data = {metric: [] for metric in metrics}
    
    for class_name in CLASS_NAMES:
        for metric in metrics:
            class_data[metric].append(report_dict[class_name][metric])
    
    x = np.arange(len(CLASS_NAMES))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    rects1 = ax.bar(x - width, class_data['precision'], width, 
                    label='Precision', color=colors[0], alpha=0.8)
    rects2 = ax.bar(x, class_data['recall'], width, 
                    label='Recall', color=colors[1], alpha=0.8)
    rects3 = ax.bar(x + width, class_data['f1-score'], width, 
                    label='F1-Score', color=colors[2], alpha=0.8)
    
    ax.set_ylabel('Score', fontsize=11)
    ax.set_xlabel('Class', fontsize=11)
    ax.set_title('Per-Class Performance Metrics', fontsize=13, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES)
    ax.legend(loc='lower right')
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(rect.get_x() + rect.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'per_class_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Per-class metrics plot saved to {output_dir / 'per_class_metrics.png'}")


def save_results(results: dict, report_dict: dict, 
                checkpoint: dict, output_path: Path):
    """Save evaluation results to JSON file."""
    summary = {
        'model_info': {
            'epochs_trained': checkpoint.get('epoch', 'unknown'),
            'best_val_accuracy': checkpoint.get('best_val_acc', 'unknown'),
        },
        'test_metrics': {
            'accuracy': float(results['accuracy']),
            'average_loss': float(results['avg_loss']),
        },
        'per_class_metrics': {
            class_name: {
                'precision': report_dict[class_name]['precision'],
                'recall': report_dict[class_name]['recall'],
                'f1_score': report_dict[class_name]['f1-score'],
                'support': int(report_dict[class_name]['support'])
            }
            for class_name in CLASS_NAMES
        },
        'overall_metrics': {
            'macro_avg_f1': report_dict['macro avg']['f1-score'],
            'weighted_avg_f1': report_dict['weighted avg']['f1-score'],
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate ankle alignment classification model')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to trained model checkpoint')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Path to processed data directory')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory to save evaluation outputs')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                       help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    model_path = args.model_path if args.model_path else MODEL_SAVE_PATH
    data_dir = Path(args.data_dir) if args.data_dir else Path(PROCESSED_DATA_DIR)
    output_dir = Path(args.output_dir) if args.output_dir else Path(OUTPUT_DIR) / 'evaluation'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not Path(model_path).exists():
        logger.error(f"Model not found at: {model_path}")
        logger.info("Please train a model first using 02-training.py")
        return
    
    model, checkpoint = load_model(model_path, device)
    
    try:
        test_loader, test_dataset = get_test_loader(data_dir, args.batch_size)
    except FileNotFoundError as e:
        logger.error(str(e))
        logger.info("Please run 01-data-preprocessing.py first")
        return
    
    results = evaluate_model(model, test_loader, device)
    
    logger.info("\nGenerating evaluation reports...")
    
    cm, cm_norm = generate_confusion_matrix(results, output_dir)
    report_dict = generate_classification_report(results)
    plot_per_class_metrics(report_dict, output_dir)
    
    save_results(results, report_dict, checkpoint, output_dir / 'evaluation_summary.json')
    
    logger.info(f"\nEvaluation complete! Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
