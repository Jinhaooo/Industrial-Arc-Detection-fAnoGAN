import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F
import matplotlib
from pathlib import Path
import json
matplotlib.use('Agg')
from configs.configs import device
from utils.detector import AnomalyDetector
from utils.dataloader import LeftCrop, ImageDataset
from utils.visualize import load_custom_image
from utils.GANtrainer import GANTrainer

def train_gan():
    
    print(" start GAN traing")
    print("="*50)
    
  
    config = {
        'nz': 100,           
        'lr_g': 0.0003,      
        'lr_d': 0.00008,     
        'beta1': 0.5,        
        'beta2': 0.999,      
        'num_epochs': 25,    
        'batch_size': 4      
    }
    

    transform = transforms.Compose([
        LeftCrop(1280),
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # path of dataset
    dataset_folder = r""
    
    
    dataset = ImageDataset(dataset_folder, transform=transform)
    
    if len(dataset) == 0:
        print(" did not find dataset")
        return
    
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=2)
    
    
    trainer = GANTrainer(config)
    
  
    training_history = trainer.train(dataloader, config['num_epochs'], save_dir="./trained_models")
    
    print("finish training ")
    return training_history

def detect_anomalies(model_dir="./trained_models", test_images_dir=None):
    
    print(" start detecting")
    print("="*50)
    
    try:
        
        detector = AnomalyDetector(model_dir)
        
        
        transform = transforms.Compose([
            LeftCrop(1280),
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        
        if test_images_dir is None:
            test_images_dir = r""
        
        if not os.path.exists(test_images_dir):
            print(f" the path of test image is not existed: {test_images_dir}")
            return
        
        test_image_paths = []
        for file in os.listdir(test_images_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                test_image_paths.append(os.path.join(test_images_dir, file))
        
        if len(test_image_paths) == 0:
            print(f" did not find test image in {test_images_dir} ")
            return
        
        print(f"find {len(test_image_paths)} test images")
        
      
        results = []
        for img_path in test_image_paths:
            print(f"\n detection: {os.path.basename(img_path)}")
            
          
            try:
                image = Image.open(img_path).convert('RGB')
                test_image = transform(image)
            except Exception as e:
                print(f" can not load images: {e}")
                continue
            
            
            score, reconstruction, details = detector.detect_anomaly(
                test_image, 
                num_iterations=300,
                method='brightness_aware'
            )
            
            results.append({
                'path': img_path,
                'name': os.path.basename(img_path),
                'score': score,
                'original': test_image,
                'reconstruction': reconstruction,
                'details': details
            })
        
    
        if results:
            save_detection_results(results)
        
        print("\n finish detection！")
        return results
        
    except Exception as e:
        print(f" errors in detection: {e}")
        return None

def save_detection_results(results, save_dir="./detection_results"):
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    

    num_results = len(results)
    fig, axes = plt.subplots(num_results, 4, figsize=(16, 4*num_results))
    
    if num_results == 1:
        axes = axes.reshape(1, -1)
    
    for i, result in enumerate(results):
        
        orig_img = result['original'].permute(1, 2, 0) * 0.5 + 0.5
        orig_img = torch.clamp(orig_img, 0, 1)
        axes[i, 0].imshow(orig_img)
        axes[i, 0].set_title(f'Original\n{result["name"]}')
        axes[i, 0].axis('off')
        
        
        recon_img = result['reconstruction'].permute(1, 2, 0).detach() * 0.5 + 0.5
        recon_img = torch.clamp(recon_img, 0, 1)
        axes[i, 1].imshow(recon_img)
        axes[i, 1].set_title(f'Reconstructed\nScore: {result["score"]:.4f}')
        axes[i, 1].axis('off')
        
    
        diff = torch.abs(result['original'] - result['reconstruction'])
        diff_img = diff.permute(1, 2, 0).detach()
        diff_img = torch.clamp(diff_img * 3, 0, 1)
        axes[i, 2].imshow(diff_img)
        axes[i, 2].set_title('Difference (Enhanced)')
        axes[i, 2].axis('off')
        
        
        diff_gray = torch.mean(diff, dim=0).detach().numpy()
        im = axes[i, 3].imshow(diff_gray, cmap='hot')
        axes[i, 3].set_title('Anomaly Heatmap')
        axes[i, 3].axis('off')
        plt.colorbar(im, ax=axes[i, 3], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    result_path = Path(save_dir) / "detection_results.png"
    plt.savefig(result_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f" save result to: {result_path}")
    
    
    results_summary = []
    for result in results:
        results_summary.append({
            'filename': result['name'],
            'anomaly_score': result['score'],
            'loss_components': result['details']
        })
    
    json_path = Path(save_dir) / "detection_summary.json"
    with open(json_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f" save result to: {json_path}")



def main():
    
    
    import argparse
    
    parser = argparse.ArgumentParser(description='GAN')
    parser.add_argument('--mode', choices=['train', 'detect', 'both'], 
                       default='both', )
    parser.add_argument('--model_dir', default='./trained_models', 
                       )
    parser.add_argument('--test_dir', default=None,
                       )
    
    args = parser.parse_args()
    
    if args.mode in ['train', 'both']:
        print("step 1: training GAN")
        train_gan()
        print("\n" + "="*50 + "\n")
    
    if args.mode in ['detect', 'both']:
        print(" step2: start detection")
        detect_anomalies(args.model_dir, args.test_dir)

if __name__ == "__main__":
    
    
    print(" GAN for detection")
    print("choose models:")
    print("1. only training")
    print("2. only detecting")
    print("3. both")
    
    choice = input("enter numbers: (1/2/3): ").strip()
    
    if choice == '1':
        print("\n start training...")
        train_gan()
        
    elif choice == '2':
        print("\n start detecting...")
      
        model_dir = './trained_models'
        
   
        test_dir = None
            
        detect_anomalies(model_dir, test_dir)
        
    elif choice == '3':
        print("\n start training...")
        train_gan()
        print("\n start detecting...")
        detect_anomalies('./trained_models')
        
    else:
        print(" invalid choice")
        
  
