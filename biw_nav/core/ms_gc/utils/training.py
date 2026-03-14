import torch
import os

def save_checkpoint(model, decoder, optimizer, epoch, checkpoint_dir):
    """保存模型检查点"""
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }, checkpoint_path)
    print(f'Checkpoint saved at {checkpoint_path}')

def load_checkpoint(model, decoder, optimizer, checkpoint_path, device):
    """加载模型检查点"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f'Checkpoint loaded from {checkpoint_path}, epoch {epoch}')
    return epoch

def train_one_epoch(trainer, train_loader, epoch):
    """训练一个epoch"""
    trainer.grid_model.train()
    trainer.decoder.train()
    
    train_loss = 0.0
    for batch_idx, batch in enumerate(train_loader):
        positions = batch['positions'].to(trainer.device)
        targets = batch['targets'].to(trainer.device)
        
        # 前向传播
        grid_activities = trainer.grid_model(positions)
        decoded_positions = trainer.decoder(grid_activities)
        
        # 计算损失
        loss = sum(trainer.criterion(decoded, targets) for decoded in decoded_positions)
        
        # 反向传播和优化
        trainer.optimizer.zero_grad()
        loss.backward()
        trainer.optimizer.step()
        
        train_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    avg_train_loss = train_loss / len(train_loader)
    print(f'Epoch {epoch+1}, Average Train Loss: {avg_train_loss:.4f}')
    return avg_train_loss

def validate(trainer, val_loader):
    """验证模型"""
    trainer.grid_model.eval()
    trainer.decoder.eval()
    
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            positions = batch['positions'].to(trainer.device)
            targets = batch['targets'].to(trainer.device)
            
            grid_activities = trainer.grid_model(positions)
            decoded_positions = trainer.decoder(grid_activities)
            
            loss = sum(trainer.criterion(decoded, targets) for decoded in decoded_positions)
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(val_loader)
    print(f'Validation Loss: {avg_val_loss:.4f}')
    return avg_val_loss