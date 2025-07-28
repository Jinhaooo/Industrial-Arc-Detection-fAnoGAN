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
matplotlib.use('Agg')
from configs.configs import device
from models.discriminator import Discriminator
from models.generator import Generator
from pathlib import Path
import json

class AnomalyDetector:
    
    
    def __init__(self, model_dir="./models"):
        self.model_dir = Path(model_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
      
        
        
        self.config = self._load_config()
        
        
        self.netG = Generator(nz=self.config['nz']).to(self.device)
        self.netD = Discriminator().to(self.device)
        
      
        self._load_models()
        
        
        self.brightness_loss = BrightnessAwareReconstructionLoss()
        
        
    
    def _load_config(self):
        """加载配置"""
        config_path = self.model_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"did not find configs: {config_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        
        return config
    
    def _load_models(self):
        
        generator_path = self.model_dir / "generator.pth"
        if not generator_path.exists():
            raise FileNotFoundError(f"did not find generator: {generator_path}")
        
        self.netG.load_state_dict(torch.load(generator_path, map_location=self.device))
        self.netG.eval()
        print(f" load generator: {generator_path}")
        
        
        discriminator_path = self.model_dir / "discriminator.pth"
        if not discriminator_path.exists():
            raise FileNotFoundError(f"did not find discriminator: {discriminator_path}")
        
        self.netD.load_state_dict(torch.load(discriminator_path, map_location=self.device))
        self.netD.eval()
        print(f" load discirminator: {discriminator_path}")
    
    def detect_anomaly(self, test_image, num_iterations=300, method='brightness_aware'):
     
        
        
        if method == 'brightness_aware':
            return self._brightness_aware_detection(test_image, num_iterations)
        elif method == 'original':
            return self._original_detection(test_image, num_iterations)
        else:
            raise ValueError(f": {method}")
    
    def _brightness_aware_detection(self, test_image, num_iterations):
      
        best_score = float('inf')
        best_reconstruction = None
        best_loss_components = None
        
        for trial in range(2):
            z = torch.randn(1, self.config['nz'], device=self.device, requires_grad=True)
            optimizer = optim.Adam([z], lr=0.02)
            
            test_batch = test_image.unsqueeze(0).to(self.device)
            
            for i in range(num_iterations):
                optimizer.zero_grad()
                
                generated = self.netG(z)
                
              
                recon_loss, loss_components = self.brightness_loss(test_batch, generated)
                
                
                disc_score = self.netD(generated)
                disc_loss = F.mse_loss(disc_score, torch.ones_like(disc_score))
                
                total_loss = recon_loss + 0.1 * disc_loss
                total_loss.backward()
                optimizer.step()
                
                if i % 100 == 0 and trial == 0:
                    print(f'   Trial {trial+1}, Iter {i}: Total: {total_loss.item():.4f}')
            
          
            with torch.no_grad():
                final_generated = self.netG(z)
                final_loss, final_components = self.brightness_loss(test_batch, final_generated)
                
                if final_loss.item() < best_score:
                    best_score = final_loss.item()
                    best_reconstruction = final_generated.squeeze(0).cpu()
                    best_loss_components = final_components
        
       
        return best_score, best_reconstruction, best_loss_components
    
    
class BrightnessAwareReconstructionLoss(nn.Module):

    
    def __init__(self, alpha=1.0, beta=2.0, gamma=1.5):
        super().__init__()
        self.alpha = alpha  # 空间梯度权重
        self.beta = beta    # 局部对比度权重  
        self.gamma = gamma  # 频域权重
        
    def spatial_gradient_loss(self, x, y):
        
        
        grad_x_real = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])
        grad_y_real = torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])
        
        grad_x_fake = torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])
        grad_y_fake = torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])
        
      
        grad_diff_x = F.mse_loss(grad_x_real, grad_x_fake)
        grad_diff_y = F.mse_loss(grad_y_real, grad_y_fake)
        
        return grad_diff_x + grad_diff_y
    
    def local_contrast_loss(self, x, y, window_size=5):
      
        kernel = torch.ones(1, 1, window_size, window_size, device=x.device) / (window_size * window_size)
        
        
        x_mean = F.conv2d(x.mean(dim=1, keepdim=True), kernel, padding=window_size//2)
        y_mean = F.conv2d(y.mean(dim=1, keepdim=True), kernel, padding=window_size//2)
        
      
        x_var = F.conv2d((x.mean(dim=1, keepdim=True) - x_mean)**2, kernel, padding=window_size//2)
        y_var = F.conv2d((y.mean(dim=1, keepdim=True) - y_mean)**2, kernel, padding=window_size//2)
        
        x_std = torch.sqrt(x_var + 1e-8)
        y_std = torch.sqrt(y_var + 1e-8)
        
        
        contrast_diff = F.mse_loss(x_std, y_std)
        brightness_diff = F.mse_loss(x_mean, y_mean)
        
        return contrast_diff + brightness_diff
    
    def frequency_domain_loss(self, x, y):
      
        
        low_freq_kernel = torch.ones(1, 1, 7, 7, device=x.device) / 49
        x_low = F.conv2d(x.mean(dim=1, keepdim=True), low_freq_kernel, padding=3)
        y_low = F.conv2d(y.mean(dim=1, keepdim=True), low_freq_kernel, padding=3)
        low_freq_loss = F.mse_loss(x_low, y_low)
        
        
        high_freq_kernel = torch.tensor([[[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]]], 
                                       dtype=torch.float32, device=x.device)
        x_high = F.conv2d(x.mean(dim=1, keepdim=True), high_freq_kernel, padding=1)
        y_high = F.conv2d(y.mean(dim=1, keepdim=True), high_freq_kernel, padding=1)
        high_freq_loss = F.mse_loss(x_high, y_high)
        
        return low_freq_loss + high_freq_loss
    
    def forward(self, x, y):
        
        
        mse_loss = F.mse_loss(x, y)
        
        
        spatial_loss = self.spatial_gradient_loss(x, y)
        
        
        contrast_loss = self.local_contrast_loss(x, y)
        
    
        freq_loss = self.frequency_domain_loss(x, y)
        
        
        total_loss = mse_loss + self.alpha * spatial_loss + self.beta * contrast_loss + self.gamma * freq_loss
        
        return total_loss, {
            'mse': mse_loss.item(),
            'spatial': spatial_loss.item(),
            'contrast': contrast_loss.item(),
            'frequency': freq_loss.item()
        }
class LocalizedAnomalyDetector:
  
    
    def __init__(self, detector, patch_size=64, stride=32):
        self.detector = detector
        self.patch_size = patch_size
        self.stride = stride
        self.brightness_loss = BrightnessAwareReconstructionLoss()
    
    def sliding_window_detection(self, image, num_iterations=200):
        "
        print(f" sliding_window_detection，size of window: {self.patch_size}x{self.patch_size}")
        
        c, h, w = image.shape
        anomaly_map = torch.zeros(h, w)
        
        
        for y in range(0, h - self.patch_size + 1, self.stride):
            for x in range(0, w - self.patch_size + 1, self.stride):
                
                patch = image[:, y:y+self.patch_size, x:x+self.patch_size]
                
                
                score, _ = self.detect_patch_anomaly(patch, num_iterations)
                
                
                anomaly_map[y:y+self.patch_size, x:x+self.patch_size] += score
        
      
        anomaly_map = anomaly_map / anomaly_map.max() if anomaly_map.max() > 0 else anomaly_map
        
        return anomaly_map
    
    def detect_patch_anomaly(self, patch, num_iterations=200):
        """检测单个patch的异常"""
        self.detector.netG.eval()
        self.detector.netD.eval()
        
        
        if patch.shape[-1] != 256:
            patch_resized = F.interpolate(patch.unsqueeze(0), size=(256, 256), mode='bilinear')
        else:
            patch_resized = patch.unsqueeze(0)
        
        patch_resized = patch_resized.to(self.detector.device)
        
        
        z = torch.randn(1, self.detector.nz, device=self.detector.device, requires_grad=True)
        optimizer = optim.Adam([z], lr=0.02)
        
        best_loss = float('inf')
        best_z = z.clone()
        
        for i in range(num_iterations):
            optimizer.zero_grad()
            
            generated = self.detector.netG(z)
            
            
            recon_loss, loss_components = self.brightness_loss(patch_resized, generated)
            
            
            disc_score = self.detector.netD(generated)
            disc_loss = F.mse_loss(disc_score, torch.ones_like(disc_score))
            
            total_loss = recon_loss + 0.1 * disc_loss
            total_loss.backward()
            optimizer.step()
            
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_z = z.clone().detach()
        
        
        with torch.no_grad():
            final_generated = self.detector.netG(best_z)
            final_loss, _ = self.brightness_loss(patch_resized, final_generated)
            
        return final_loss.item(), final_generated.squeeze(0).cpu()

def enhanced_anomaly_detection(detector, test_image, method='comprehensive'):
    
    
    if method == 'brightness_aware':
        return brightness_aware_detection(detector, test_image)
    elif method == 'localized':
        return localized_detection(detector, test_image)
    elif method == 'comprehensive':
        return comprehensive_detection(detector, test_image)
    else:
        raise ValueError("Unknown detection method")

def brightness_aware_detection(detector, test_image, num_iterations=300):
    
    print(" brightness_aware_detection...")
    
    detector.netG.eval()
    detector.netD.eval()
    
    brightness_loss = BrightnessAwareReconstructionLoss()
    
    
    best_score = float('inf')
    best_reconstruction = None
    best_loss_components = None
    
    for trial in range(2):
        z = torch.randn(1, detector.nz, device=detector.device, requires_grad=True)
        optimizer = optim.Adam([z], lr=0.02)
        
        test_batch = test_image.unsqueeze(0).to(detector.device)
        
        for i in range(num_iterations):
            optimizer.zero_grad()
            
            generated = detector.netG(z)
            
            
            recon_loss, loss_components = brightness_loss(test_batch, generated)
            
            
            disc_score = detector.netD(generated)
            disc_loss = F.mse_loss(disc_score, torch.ones_like(disc_score))
            
            total_loss = recon_loss + 0.1 * disc_loss
            total_loss.backward()
            optimizer.step()
            
            if i % 100 == 0 and trial == 0:
                print(f'Iteration {i}: Total: {total_loss.item():.4f}, '
                      f'MSE: {loss_components["mse"]:.4f}, '
                      f'Spatial: {loss_components["spatial"]:.4f}, '
                      f'Contrast: {loss_components["contrast"]:.4f}')
        
      
        with torch.no_grad():
            final_generated = detector.netG(z)
            final_loss, final_components = brightness_loss(test_batch, final_generated)
            
            if final_loss.item() < best_score:
                best_score = final_loss.item()
                best_reconstruction = final_generated.squeeze(0).cpu()
                best_loss_components = final_components
    
    print(f" 最佳异常分数: {best_score:.4f}")
    print(f"   - MSE损失: {best_loss_components['mse']:.4f}")
    print(f"   - 空间梯度: {best_loss_components['spatial']:.4f}")
    print(f"   - 局部对比度: {best_loss_components['contrast']:.4f}")
    print(f"   - 频域损失: {best_loss_components['frequency']:.4f}")
    
    return best_score, best_reconstruction, best_loss_components

def localized_detection(detector, test_image):
    
    print(" localized_detection...")
    
    localizer = LocalizedAnomalyDetector(detector, patch_size=64, stride=32)
    anomaly_map = localizer.sliding_window_detection(test_image)
    
    # 计算全局异常分数
    global_score = torch.mean(anomaly_map).item()
    
    return global_score, anomaly_map

def comprehensive_detection(detector, test_image):
    
    print(" comprehensive_detection...")
    
    
    brightness_score, brightness_recon, loss_components = brightness_aware_detection(detector, test_image)
    
      
    local_score, anomaly_map = localized_detection(detector, test_image)
    
    
    combined_score = 0.7 * brightness_score + 0.3 * local_score
    
    return {
        'combined_score': combined_score,
        'brightness_score': brightness_score,
        'local_score': local_score,
        'reconstruction': brightness_recon,
        'anomaly_map': anomaly_map,
        'loss_components': loss_components
    }



def visualize_brightness_detection_results(results, save_path="brightness_detection_results.png"):
  
    
    if isinstance(results, dict) and 'combined_score' in results:
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
      
        orig_img = results['original'].permute(1, 2, 0) * 0.5 + 0.5
        orig_img = torch.clamp(orig_img, 0, 1)
        axes[0, 0].imshow(orig_img)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        
        recon_img = results['reconstruction'].permute(1, 2, 0) * 0.5 + 0.5
        recon_img = torch.clamp(recon_img, 0, 1)
        axes[0, 1].imshow(recon_img)
        axes[0, 1].set_title(f'Reconstruction\nScore: {results["brightness_score"]:.4f}')
        axes[0, 1].axis('off')
        
        
        diff = torch.abs(results['original'] - results['reconstruction'])
        diff_img = diff.permute(1, 2, 0)
        diff_img = torch.clamp(diff_img * 3, 0, 1)
        axes[0, 2].imshow(diff_img)
        axes[0, 2].set_title('Pixel Difference')
        axes[0, 2].axis('off')
        
      
        im1 = axes[1, 0].imshow(results['anomaly_map'], cmap='hot')
        axes[1, 0].set_title(f'Localized Anomaly Map\nScore: {results["local_score"]:.4f}')
        axes[1, 0].axis('off')
        plt.colorbar(im1, ax=axes[1, 0])
        
        
        components = results['loss_components']
        comp_names = list(components.keys())
        comp_values = list(components.values())
        
        axes[1, 1].bar(comp_names, comp_values, color=['blue', 'orange', 'green', 'red'])
        axes[1, 1].set_title('Loss Components Analysis')
        axes[1, 1].set_ylabel('Loss Value')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
      
        axes[1, 2].text(0.5, 0.7, f'Combined Score:', ha='center', va='center', 
                        fontsize=16, weight='bold', transform=axes[1, 2].transAxes)
        axes[1, 2].text(0.5, 0.5, f'{results["combined_score"]:.4f}', ha='center', va='center',
                        fontsize=24, color='red', weight='bold', transform=axes[1, 2].transAxes)
        
        axes[1, 2].text(0.5, 0.3, f'Brightness: {results["brightness_score"]:.4f}\nLocal: {results["local_score"]:.4f}',
                        ha='center', va='center', fontsize=12, transform=axes[1, 2].transAxes)
        axes[1, 2].set_xlim(0, 1)
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].axis('off')
        
    else:
        
        num_results = len(results)
        fig, axes = plt.subplots(num_results, 4, figsize=(16, 4*num_results))
        
        if num_results == 1:
            axes = axes.reshape(1, -1)
        
        for i, result in enumerate(results):
            
            orig_img = result['original'].permute(1, 2, 0) * 0.5 + 0.5
            orig_img = torch.clamp(orig_img, 0, 1)
            axes[i, 0].imshow(orig_img)
            axes[i, 0].set_title(f'Original: {result["name"]}')
            axes[i, 0].axis('off')
            
            
            recon_img = result['reconstruction'].permute(1, 2, 0) * 0.5 + 0.5
            recon_img = torch.clamp(recon_img, 0, 1)
            axes[i, 1].imshow(recon_img)
            axes[i, 1].set_title(f'Reconstructed\nScore: {result["score"]:.4f}')
            axes[i, 1].axis('off')
            
            
            diff = torch.abs(result['original'] - result['reconstruction'])
            diff_img = diff.permute(1, 2, 0)
            diff_img = torch.clamp(diff_img * 3, 0, 1)
            axes[i, 2].imshow(diff_img)
            axes[i, 2].set_title('Difference')
            axes[i, 2].axis('off')
            
            
            diff_gray = torch.mean(diff, dim=0)
            im = axes[i, 3].imshow(diff_gray, cmap='hot')
            axes[i, 3].set_title('Anomaly Heatmap')
            axes[i, 3].axis('off')
            plt.colorbar(im, ax=axes[i, 3])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"save anomaly detection result to: {save_path}")


