#!/usr/bin/env python3
"""
Batch prediction script for testing multiple histopathological images
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import glob
import pandas as pd
from datetime import datetime

# Import the CNN model from app.py
from app import CNN, device, model, transform

def predict_batch(image_folder, output_file=None):
    """
    Predict cancer probability for all images in a folder
    
    Args:
        image_folder (str): Path to folder containing images
        output_file (str): Optional CSV file to save results
    """
    
    # Supported image extensions
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']
    
    # Find all images in the folder
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(image_folder, ext)))
        image_files.extend(glob.glob(os.path.join(image_folder, ext.upper())))
    
    if not image_files:
        print(f"No images found in {image_folder}")
        return
    
    print(f"Found {len(image_files)} images to process...")
    
    results = []
    
    for i, image_path in enumerate(image_files, 1):
        try:
            print(f"Processing {i}/{len(image_files)}: {os.path.basename(image_path)}")
            
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = transform(image)
            image_tensor = image_tensor.unsqueeze(0).to(device)
            
            # Make prediction
            with torch.no_grad():
                output = model(image_tensor)
                probability = output.item()
                prediction = 1 if probability > 0.5 else 0
            
            # Determine result and confidence
            if prediction == 1:
                result = "Cancer Detected"
                confidence = "High" if probability > 0.8 else "Medium" if probability > 0.6 else "Low"
            else:
                result = "No Cancer Detected"
                confidence = "High" if probability < 0.2 else "Medium" if probability < 0.4 else "Low"
            
            # Store results
            results.append({
                'filename': os.path.basename(image_path),
                'filepath': image_path,
                'prediction': prediction,
                'probability': round(probability, 4),
                'result': result,
                'confidence': confidence,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            
            print(f"  Result: {result} (Confidence: {confidence}, Probability: {probability:.4f})")
            
        except Exception as e:
            print(f"  Error processing {image_path}: {e}")
            results.append({
                'filename': os.path.basename(image_path),
                'filepath': image_path,
                'prediction': None,
                'probability': None,
                'result': f"Error: {str(e)}",
                'confidence': None,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Print summary
    print("\n" + "="*60)
    print("BATCH PREDICTION SUMMARY")
    print("="*60)
    
    successful_predictions = df[df['prediction'].notna()]
    if len(successful_predictions) > 0:
        cancer_detected = len(successful_predictions[successful_predictions['prediction'] == 1])
        no_cancer = len(successful_predictions[successful_predictions['prediction'] == 0])
        
        print(f"Total images processed: {len(df)}")
        print(f"Successful predictions: {len(successful_predictions)}")
        print(f"Cancer detected: {cancer_detected}")
        print(f"No cancer detected: {no_cancer}")
        print(f"Cancer detection rate: {cancer_detected/len(successful_predictions)*100:.1f}%")
        
        if len(successful_predictions) > 0:
            avg_probability = successful_predictions['probability'].mean()
            print(f"Average probability: {avg_probability:.4f}")
    
    # Save results to CSV if output file specified
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
    
    return df

def main():
    """Main function"""
    print("="*60)
    print("Histopathological Cancer Detection - Batch Prediction")
    print("="*60)
    
    # Check if model is loaded
    if model is None:
        print("❌ Model not loaded. Please run the Flask app first or check model.pt")
        return
    
    # Get input folder
    image_folder = input("Enter the path to the folder containing images: ").strip()
    
    if not os.path.exists(image_folder):
        print(f"❌ Folder not found: {image_folder}")
        return
    
    # Get output file
    output_file = input("Enter output CSV file path (or press Enter to skip): ").strip()
    if not output_file:
        output_file = None
    
    # Run batch prediction
    try:
        results_df = predict_batch(image_folder, output_file)
        print("\n✅ Batch prediction completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during batch prediction: {e}")

if __name__ == "__main__":
    main() 