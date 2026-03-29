import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from dataset import get_dataloaders
from model import ViolenceClassifier
import time

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # GPU-specific optimizations
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True  # cuDNN auto-tuner for better performance

    # Optimized batch size (increased for faster training)
    # If you run out of memory, reduce to 64 or 48
    batch_size = 96

    train_loader, val_loader = get_dataloaders('../Violence_Detection.v1i.multiclass', batch_size=batch_size)

    # Create enhanced model with pre-trained weights
    model = ViolenceClassifier(num_classes=4, use_pretrained=True).to(device)

    # Enable gradient checkpointing to save memory (if using GPU)
    if device.type == 'cuda':
        # This allows larger batch sizes with similar memory usage
        for module in model.backbone.modules():
            if hasattr(module, 'gradient_checkpointing'):
                module.gradient_checkpointing = True
    
    # Loss function for multi-label classification
    criterion = nn.BCEWithLogitsLoss()
    
    # Optimizer with optimized parameters
    optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4, betas=(0.9, 0.999))
    
    # Combined scheduler strategy: Cosine annealing with warm restarts + ReduceLROnPlateau
    scheduler_warmrestart = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)
    scheduler_reduce = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4, threshold=1e-4)
    
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 10  # Early stopping if no improvement for 10 epochs
    
    epochs = 30  # Reduced epochs with early stopping as safety
    
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        total_train_loss = 0
        batch_count = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_train_loss += loss.item()
            batch_count += 1
            
            if (batch_idx + 1) % 10 == 0:
                avg_batch_loss = total_train_loss / batch_count
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}, Avg: {avg_batch_loss:.4f}")
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        epoch_time = time.time() - epoch_start_time
        
        print(f"Epoch {epoch+1}/{epochs} ({epoch_time:.2f}s)")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}")
        print("-" * 50)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'violence_model.pth')
            print(f"  ✓ Model saved! (Best Val Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  ⚠ No improvement ({patience_counter}/{max_patience})")
        
        # Early stopping
        if patience_counter >= max_patience:
            print(f"\nEarly stopping triggered! No improvement for {max_patience} epochs.")
            break
        
        # Update learning rates
        scheduler_warmrestart.step()
        scheduler_reduce.step(avg_val_loss)

        # Clear GPU cache per epoch to manage memory
        if device.type == "cuda":
            torch.cuda.empty_cache()
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.2f}s!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("Model saved as 'violence_model.pth'")

if __name__ == '__main__':
    train()