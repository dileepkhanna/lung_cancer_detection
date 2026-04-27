"""Evaluation script for trained model"""

import os
import torch
import numpy as np
import pickle
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import config
from model import get_model
from train import LIDCDataset


def plot_confusion_matrix(cm, save_path):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()


def plot_roc_curve(fpr, tpr, auc_score, save_path):
    """Plot and save ROC curve"""
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def evaluate_model():
    """Evaluate trained model on test set"""
    print("Loading preprocessed data...")
    data_file = os.path.join(config.PROCESSED_DATA_DIR, 'processed_data.pkl')
    with open(data_file, 'rb') as f:
        all_data = pickle.load(f)
    
    # Create dataset and split
    dataset = LIDCDataset(all_data)
    train_size = int(config.TRAIN_SPLIT * len(dataset))
    val_size = int(config.VAL_SPLIT * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    _, _, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config.RANDOM_SEED)
    )
    
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = get_model(config.MODEL_NAME, config.NUM_CLASSES, config.PRETRAINED)
    checkpoint = torch.load(os.path.join(config.MODEL_DIR, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Evaluate
    all_preds = []
    all_labels = []
    all_probs = []
    all_patient_ids = []
    all_file_paths = []
    
    print("Evaluating on test set...")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    # Get patient IDs and file paths from test dataset
    test_indices = test_dataset.indices
    for idx in test_indices:
        all_patient_ids.append(all_data[idx]['patient_id'])
        all_file_paths.append(all_data[idx]['file_path'])
    
    # Calculate metrics
    print("\nClassification Report:")
    report = classification_report(all_labels, all_preds, target_names=['No Nodule', 'Nodule'], output_dict=True)
    print(classification_report(all_labels, all_preds, target_names=['No Nodule', 'Nodule']))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, os.path.join(config.MODEL_DIR, 'confusion_matrix.png'))
    print("\nConfusion Matrix saved")
    
    # ROC curve
    auc_score = 0
    if len(np.unique(all_labels)) == 2:
        auc_score = roc_auc_score(all_labels, all_probs)
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        plot_roc_curve(fpr, tpr, auc_score, os.path.join(config.MODEL_DIR, 'roc_curve.png'))
        print(f"AUC Score: {auc_score:.4f}")
        print("ROC Curve saved")
    
    # Calculate accuracy
    accuracy = 100. * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    print(f"\nTest Accuracy: {accuracy:.2f}%")
    
    # Save detailed results to CSV
    print("\nSaving results to CSV files...")
    save_results_to_csv(all_labels, all_preds, all_probs, all_patient_ids, all_file_paths, report, accuracy, auc_score)
    
    return accuracy, auc_score


def save_results_to_csv(labels, predictions, probabilities, patient_ids, file_paths, report, accuracy, auc_score):
    """Save evaluation results to CSV files"""
    import pandas as pd
    from datetime import datetime
    
    # 1. Detailed predictions CSV
    predictions_df = pd.DataFrame({
        'patient_id': patient_ids,
        'file_path': file_paths,
        'true_label': labels,
        'predicted_label': predictions,
        'probability_nodule': probabilities,
        'correct': [1 if l == p else 0 for l, p in zip(labels, predictions)]
    })
    
    predictions_csv = os.path.join(config.MODEL_DIR, 'predictions.csv')
    predictions_df.to_csv(predictions_csv, index=False)
    print(f"✓ Saved predictions: {predictions_csv}")
    
    # 2. Summary metrics CSV
    summary_data = {
        'metric': ['Accuracy', 'AUC-ROC', 'Precision (No Nodule)', 'Recall (No Nodule)', 
                   'F1-Score (No Nodule)', 'Precision (Nodule)', 'Recall (Nodule)', 
                   'F1-Score (Nodule)', 'Total Samples', 'Correct Predictions', 'Wrong Predictions'],
        'value': [
            f"{accuracy:.2f}%",
            f"{auc_score:.4f}",
            f"{report['No Nodule']['precision']:.4f}",
            f"{report['No Nodule']['recall']:.4f}",
            f"{report['No Nodule']['f1-score']:.4f}",
            f"{report['Nodule']['precision']:.4f}",
            f"{report['Nodule']['recall']:.4f}",
            f"{report['Nodule']['f1-score']:.4f}",
            len(labels),
            sum([1 if l == p else 0 for l, p in zip(labels, predictions)]),
            sum([1 if l != p else 0 for l, p in zip(labels, predictions)])
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_csv = os.path.join(config.MODEL_DIR, 'evaluation_summary.csv')
    summary_df.to_csv(summary_csv, index=False)
    print(f"✓ Saved summary: {summary_csv}")
    
    # 3. Per-patient summary CSV
    patient_summary = predictions_df.groupby('patient_id').agg({
        'true_label': 'first',
        'predicted_label': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0],
        'probability_nodule': 'mean',
        'correct': 'mean'
    }).reset_index()
    
    patient_summary.columns = ['patient_id', 'true_label', 'predicted_label', 'avg_probability', 'accuracy']
    patient_summary['accuracy'] = patient_summary['accuracy'] * 100
    
    patient_csv = os.path.join(config.MODEL_DIR, 'patient_summary.csv')
    patient_summary.to_csv(patient_csv, index=False)
    print(f"✓ Saved patient summary: {patient_csv}")
    
    # 4. Training history CSV (if exists)
    log_dir = config.LOG_DIR
    if os.path.exists(log_dir):
        try:
            from tensorboard.backend.event_processing import event_accumulator
            ea = event_accumulator.EventAccumulator(log_dir)
            ea.Reload()
            
            # Extract training metrics
            train_loss = ea.Scalars('Loss/train')
            val_loss = ea.Scalars('Loss/val')
            train_acc = ea.Scalars('Accuracy/train')
            val_acc = ea.Scalars('Accuracy/val')
            
            history_df = pd.DataFrame({
                'epoch': range(len(train_loss)),
                'train_loss': [x.value for x in train_loss],
                'val_loss': [x.value for x in val_loss],
                'train_accuracy': [x.value for x in train_acc],
                'val_accuracy': [x.value for x in val_acc]
            })
            
            history_csv = os.path.join(config.MODEL_DIR, 'training_history.csv')
            history_df.to_csv(history_csv, index=False)
            print(f"✓ Saved training history: {history_csv}")
        except:
            print("  (Training history not available)")
    
    # 5. Model configuration CSV
    config_data = {
        'parameter': ['Model', 'Image Size', 'Batch Size', 'Learning Rate', 'Epochs', 
                     'Train Split', 'Val Split', 'Test Split', 'Dropout Rate', 'Liquid Steps'],
        'value': [
            config.MODEL_NAME,
            str(config.IMAGE_SIZE),
            config.BATCH_SIZE,
            config.LEARNING_RATE,
            config.NUM_EPOCHS,
            config.TRAIN_SPLIT,
            config.VAL_SPLIT,
            config.TEST_SPLIT,
            config.DROPOUT_RATE,
            config.NUM_LIQUID_STEPS if hasattr(config, 'NUM_LIQUID_STEPS') else 'N/A'
        ]
    }
    
    config_df = pd.DataFrame(config_data)
    config_csv = os.path.join(config.MODEL_DIR, 'model_configuration.csv')
    config_df.to_csv(config_csv, index=False)
    print(f"✓ Saved configuration: {config_csv}")
    
    # 6. Timestamp and metadata
    metadata = {
        'info': ['Evaluation Date', 'Model File', 'Dataset', 'Device'],
        'value': [
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'best_model.pth',
            'LIDC-IDRI',
            'GPU' if torch.cuda.is_available() else 'CPU'
        ]
    }
    
    metadata_df = pd.DataFrame(metadata)
    metadata_csv = os.path.join(config.MODEL_DIR, 'metadata.csv')
    metadata_df.to_csv(metadata_csv, index=False)
    print(f"✓ Saved metadata: {metadata_csv}")
    
    print(f"\n✓ All CSV files saved in: {config.MODEL_DIR}/")
    print("\nGenerated CSV files:")
    print("  1. predictions.csv - Detailed predictions for each image")
    print("  2. evaluation_summary.csv - Overall metrics")
    print("  3. patient_summary.csv - Per-patient results")
    print("  4. training_history.csv - Training progress (if available)")
    print("  5. model_configuration.csv - Model settings")
    print("  6. metadata.csv - Evaluation metadata")


if __name__ == '__main__':
    evaluate_model()
