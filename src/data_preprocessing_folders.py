"""Data preprocessing with folder structure organization (cancer/normal)"""

import os
import numpy as np
import pandas as pd
import pydicom
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm
import pickle
import cv2
import config
from utils import preprocess_ct_slice


def parse_xml_annotations(xml_path: str):
    """Parse XML annotation file to extract nodule information"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    nodules = []
    for reading_session in root.findall('.//readingSession'):
        for nodule in reading_session.findall('.//unblindedReadNodule'):
            nodule_id = nodule.get('noduleID')
            
            # Extract characteristics
            characteristics = nodule.find('characteristics')
            if characteristics is not None:
                malignancy = characteristics.find('malignancy')
                if malignancy is not None:
                    nodules.append({
                        'nodule_id': nodule_id,
                        'malignancy': int(malignancy.text)
                    })
    
    return nodules


def load_patient_data(patient_dir: Path):
    """Load all DICOM files and annotations for a patient"""
    patient_data = []
    
    # Find all series directories
    for study_dir in patient_dir.iterdir():
        if not study_dir.is_dir():
            continue
            
        for series_dir in study_dir.iterdir():
            if not series_dir.is_dir():
                continue
            
            # Find XML annotation file
            xml_files = list(series_dir.glob('*.xml'))
            annotations = []
            if xml_files:
                annotations = parse_xml_annotations(str(xml_files[0]))
            
            # Load DICOM files
            dcm_files = sorted(series_dir.glob('*.dcm'))
            for dcm_file in dcm_files:
                try:
                    dicom = pydicom.dcmread(str(dcm_file))
                    image = dicom.pixel_array
                    
                    # Preprocess
                    processed_image = preprocess_ct_slice(image)
                    
                    # Determine label (simplified: any nodule = positive)
                    label = 1 if annotations else 0
                    
                    patient_data.append({
                        'image': processed_image,
                        'label': label,
                        'patient_id': patient_dir.name,
                        'file_path': str(dcm_file)
                    })
                except Exception as e:
                    print(f"Error loading {dcm_file}: {e}")
                    continue
    
    return patient_data


def create_dataset_pickle():
    """Create preprocessed dataset as pickle file (original method)"""
    print("="*60)
    print("METHOD 1: Creating pickle file")
    print("="*60)
    
    data_dir = Path(config.DATA_DIR)
    all_data = []
    
    # Get all patient directories
    patient_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    print(f"Found {len(patient_dirs)} patients")
    
    # Process each patient
    for patient_dir in tqdm(patient_dirs[:10], desc="Processing patients"):
        patient_data = load_patient_data(patient_dir)
        all_data.extend(patient_data)
    
    print(f"Total slices processed: {len(all_data)}")
    
    # Save processed data
    output_file = os.path.join(config.PROCESSED_DATA_DIR, 'processed_data.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(all_data, f)
    
    print(f"Processed data saved to {output_file}")
    
    # Create summary
    df = pd.DataFrame([{'patient_id': d['patient_id'], 'label': d['label']} for d in all_data])
    print("\nDataset Summary:")
    print(df['label'].value_counts())
    
    return all_data


def create_dataset_folders():
    """Create preprocessed dataset organized in folders (cancer/normal)"""
    print("\n" + "="*60)
    print("METHOD 2: Creating folder structure (cancer/normal)")
    print("="*60)
    
    data_dir = Path(config.DATA_DIR)
    
    # Create output directories
    output_base = Path('data/organized')
    cancer_dir = output_base / 'cancer'
    normal_dir = output_base / 'normal'
    
    cancer_dir.mkdir(parents=True, exist_ok=True)
    normal_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directories created:")
    print(f"  Cancer: {cancer_dir}")
    print(f"  Normal: {normal_dir}")
    
    # Get all patient directories
    patient_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    print(f"\nFound {len(patient_dirs)} patients")
    
    cancer_count = 0
    normal_count = 0
    
    # Process each patient
    for patient_dir in tqdm(patient_dirs[:10], desc="Processing patients"):
        
        # Find all series directories
        for study_dir in patient_dir.iterdir():
            if not study_dir.is_dir():
                continue
                
            for series_dir in study_dir.iterdir():
                if not series_dir.is_dir():
                    continue
                
                # Find XML annotation file
                xml_files = list(series_dir.glob('*.xml'))
                has_nodule = False
                if xml_files:
                    annotations = parse_xml_annotations(str(xml_files[0]))
                    has_nodule = len(annotations) > 0
                
                # Load DICOM files
                dcm_files = sorted(series_dir.glob('*.dcm'))
                for dcm_file in dcm_files:
                    try:
                        dicom = pydicom.dcmread(str(dcm_file))
                        image = dicom.pixel_array
                        
                        # Preprocess
                        processed_image = preprocess_ct_slice(image)
                        
                        # Convert to uint8 for saving as image
                        image_uint8 = (processed_image * 255).astype(np.uint8)
                        
                        # Determine output directory
                        if has_nodule:
                            output_dir = cancer_dir
                            label = 'cancer'
                            cancer_count += 1
                        else:
                            output_dir = normal_dir
                            label = 'normal'
                            normal_count += 1
                        
                        # Create unique filename
                        filename = f"{patient_dir.name}_{dcm_file.stem}.png"
                        output_path = output_dir / filename
                        
                        # Save as PNG
                        cv2.imwrite(str(output_path), image_uint8)
                        
                    except Exception as e:
                        print(f"Error processing {dcm_file}: {e}")
                        continue
    
    print(f"\n✓ Processing complete!")
    print(f"\nDataset Summary:")
    print(f"  Cancer images: {cancer_count}")
    print(f"  Normal images: {normal_count}")
    print(f"  Total images: {cancer_count + normal_count}")
    print(f"\nFolder structure:")
    print(f"  {output_base}/")
    print(f"  ├── cancer/ ({cancer_count} images)")
    print(f"  └── normal/ ({normal_count} images)")
    
    # Create metadata CSV
    metadata_file = output_base / 'metadata.csv'
    metadata = {
        'total_images': cancer_count + normal_count,
        'cancer_images': cancer_count,
        'normal_images': normal_count,
        'cancer_percentage': 100 * cancer_count / (cancer_count + normal_count),
        'normal_percentage': 100 * normal_count / (cancer_count + normal_count)
    }
    
    pd.DataFrame([metadata]).to_csv(metadata_file, index=False)
    print(f"\nMetadata saved to: {metadata_file}")
    
    return cancer_count, normal_count


def create_both_formats():
    """Create both pickle and folder formats"""
    print("\n" + "="*60)
    print("CREATING BOTH FORMATS")
    print("="*60)
    
    # Method 1: Pickle file
    all_data = create_dataset_pickle()
    
    # Method 2: Folder structure
    cancer_count, normal_count = create_dataset_folders()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("\n✓ Created pickle format:")
    print(f"  Location: data/processed/processed_data.pkl")
    print(f"  Total samples: {len(all_data)}")
    
    print("\n✓ Created folder format:")
    print(f"  Location: data/organized/")
    print(f"  Cancer: {cancer_count} images")
    print(f"  Normal: {normal_count} images")
    
    print("\n" + "="*60)
    print("USAGE")
    print("="*60)
    print("\nFor pickle format (current training script):")
    print("  python src/train.py")
    
    print("\nFor folder format (PyTorch ImageFolder):")
    print("  Use train_from_folders.py (see example below)")
    
    return all_data, cancer_count, normal_count


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess LIDC-IDRI data')
    parser.add_argument('--format', type=str, default='both',
                       choices=['pickle', 'folders', 'both'],
                       help='Output format: pickle, folders, or both')
    
    args = parser.parse_args()
    
    if args.format == 'pickle':
        create_dataset_pickle()
    elif args.format == 'folders':
        create_dataset_folders()
    else:
        create_both_formats()
