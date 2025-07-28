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
from pathlib import Path
import json
from models.discriminator import Discriminator
from models.generator import Generator


class GANTrainer:
    
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        

        
      
        self.netG = Generator(nz=config['nz']).to(self.device)
        self.netD = Discriminator().to(self.device)
        
        
        self.netG.apply(self._weights_init)
        self.netD.apply(self._weights_init)
        
        
        self.criterion = nn.BCELoss()
        self.optimizerD = optim.Adam(
            self.netD.parameters(), 
            lr=config['lr_d'], 
            betas=(config['beta1'], config['beta2']),
            weight_decay=1e-4
        )
        self.optimizerG = optim.Adam(
            self.netG.parameters(), 
            lr=config['lr_g'], 
            betas=(config['beta1'], config['beta2']),
            weight_decay=1e-5
        )
        
        
        self.schedulerG = optim.lr_scheduler.ExponentialLR(self.optimizerG, gamma=0.99)
        self.schedulerD = optim.lr_scheduler.ExponentialLR(self.optimizerD, gamma=0.995)
        
        
        self.fixed_noise = torch.randn(8, config['nz'], device=self.device)
        
    def _weights_init(self, m):
        
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if hasattr(m, 'weight'):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            if hasattr(m, 'weight'):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
            if hasattr(m, 'bias'):
                nn.init.constant_(m.bias.data, 0)
        elif classname.find('Linear') != -1:
            if hasattr(m, 'weight'):
                nn.init.xavier_normal_(m.weight.data)
            if hasattr(m, 'bias'):
                nn.init.constant_(m.bias.data, 0)
    
    def train(self, dataloader, num_epochs, save_dir="./models"):
      
        print(f"start traing，numbers of epochs: {num_epochs} ")
        
        # 创建保存目录
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        real_label = 1.0
        fake_label = 0.0
        
        # 训练历史
        training_history = {
            'G_losses': [],
            'D_losses': [],
            'D_real_scores': [],
            'D_fake_scores': []
        }
        
        for epoch in range(num_epochs):
            epoch_g_loss = 0
            epoch_d_loss = 0
            epoch_d_real = 0
            epoch_d_fake = 0
            num_batches = 0
            
            for i, data in enumerate(dataloader, 0):
                batch_size = data.size(0)
                real_data = data.to(self.device)
                
              
                self.netD.zero_grad()
                
                
                real_label_smooth = real_label - 0.1 + 0.1 * torch.rand(batch_size, device=self.device)
                output = self.netD(real_data)
                errD_real = self.criterion(output, real_label_smooth)
                errD_real.backward()
                D_x = output.mean().item()
                
                
                noise = torch.randn(batch_size, self.config['nz'], device=self.device)
                fake = self.netG(noise)
                fake_label_smooth = fake_label + 0.1 * torch.rand(batch_size, device=self.device)
                output = self.netD(fake.detach())
                errD_fake = self.criterion(output, fake_label_smooth)
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                
                errD = errD_real + errD_fake
                
              
                if D_x < 0.8 and D_G_z1 > 0.2:
                    self.optimizerD.step()
                
                
                for _ in range(3):  
                    self.netG.zero_grad()
                    
                    noise = torch.randn(batch_size, self.config['nz'], device=self.device)
                    fake = self.netG(noise)
                    
                    real_label_smooth = real_label - 0.05 + 0.05 * torch.rand(batch_size, device=self.device)
                    output = self.netD(fake)
                    errG = self.criterion(output, real_label_smooth)
                    errG.backward()
                    self.optimizerG.step()
                    
                    D_G_z2 = output.mean().item()
                
              
                epoch_g_loss += errG.item()
                epoch_d_loss += errD.item()
                epoch_d_real += D_x
                epoch_d_fake += D_G_z1
                num_batches += 1
                
                
                if i % 15 == 0:
                    print(f'[{epoch+1}/{num_epochs}][{i}/{len(dataloader)}] '
                          f'Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} '
                          f'D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f}')
            
          
            training_history['G_losses'].append(epoch_g_loss / num_batches)
            training_history['D_losses'].append(epoch_d_loss / num_batches)
            training_history['D_real_scores'].append(epoch_d_real / num_batches)
            training_history['D_fake_scores'].append(epoch_d_fake / num_batches)
            
            
            self.schedulerG.step()
            self.schedulerD.step()
            
          
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch + 1, save_dir)
                self._save_sample_images(epoch + 1, save_dir)
            
          
            if (epoch + 1) % 10 == 0:
                print(f"\n Epoch {epoch+1} 统计:")
                print(f"   G Loss: {training_history['G_losses'][-1]:.4f}")
                print(f"   D Loss: {training_history['D_losses'][-1]:.4f}")
                print(f"   D(real): {training_history['D_real_scores'][-1]:.4f}")
                print(f"   D(fake): {training_history['D_fake_scores'][-1]:.4f}")
        
        
        self.save_final_model(save_dir, training_history)
        print(f" finish traing！save model to {save_dir}")
        
        return training_history
    
    def save_checkpoint(self, epoch, save_dir):
        
        checkpoint_path = Path(save_dir) / f"checkpoint_epoch_{epoch}.pth"
        torch.save({
            'epoch': epoch,
            'generator_state_dict': self.netG.state_dict(),
            'discriminator_state_dict': self.netD.state_dict(),
            'optimizer_g_state_dict': self.optimizerG.state_dict(),
            'optimizer_d_state_dict': self.optimizerD.state_dict(),
            'config': self.config
        }, checkpoint_path)
        print(f" save check point to: {checkpoint_path}")
    
    def save_final_model(self, save_dir, training_history):
        
        save_dir = Path(save_dir)
        
        
        torch.save(self.netG.state_dict(), save_dir / "generator.pth")
        
        
        torch.save(self.netD.state_dict(), save_dir / "discriminator.pth")
        
    
        with open(save_dir / "config.json", 'w') as f:
            json.dump(self.config, f, indent=2)
        
      
        with open(save_dir / "training_history.json", 'w') as f:
            json.dump(training_history, f, indent=2)
        
        print(f" save final model:")
        print(f"   generator: {save_dir / 'generator.pth'}")
        print(f"   discriminator: {save_dir / 'discriminator.pth'}")
        print(f"   configs: {save_dir / 'config.json'}")
    
    def _save_sample_images(self, epoch, save_dir):
        
        self.netG.eval()
        with torch.no_grad():
            fake_images = self.netG(self.fixed_noise)
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(f'Training Samples - Epoch {epoch}', fontsize=16)
        
        for i in range(8):
            row = i // 4
            col = i % 4
            
            img = fake_images[i].cpu().permute(1, 2, 0) * 0.5 + 0.5
            img = torch.clamp(img, 0, 1)
            
            axes[row, col].imshow(img)
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(Path(save_dir) / f"samples_epoch_{epoch}.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        self.netG.train()
