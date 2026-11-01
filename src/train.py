import os
import json
import time
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim import AdamW

from src.data_loader import get_dataloaders
from src.model import JointBiLSTMAttention
from src.metrics import evaluate_model
from src.utils import set_seed

def train():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with open('params.json', 'r') as f:
        params = json.load(f)
        
    os.makedirs('experiments/baseline', exist_ok=True)
    
    # 1. Load Data (Adapted for Person A's exact signature)
    loaders, vocab, embedding_matrix = get_dataloaders(
        data_path='1.0/data/en-US.jsonl',
        batch_size=params.get('batch_size', 32),
        embedding_dim=params.get('embedding_dim', 300)
    )
    
    train_loader = loaders['train']
    dev_loader = loaders['dev']
    test_loader = loaders['test']
    id2slot = {v: k for k, v in vocab.slot2id.items()}
    
    # 2. Initialize Model (Adapted for Person B's exact signature)
    model = JointBiLSTMAttention(
        vocab_size=len(vocab.word2id),
        embedding_dim=params.get('embedding_dim', 300),
        hidden_dim=params.get('hidden_dim', 256),
        num_intents=len(vocab.intent2id),
        num_slots=len(vocab.slot2id),
        dropout=params.get('dropout', 0.3),
        slot_loss_weight=params.get('slot_loss_weight', 1.0),
        embedding_matrix=embedding_matrix,
        id2slot=id2slot
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params <= 20000000, f"Parameter count {total_params} exceeds 20M limit!"
    print(f"Total Model Parameters: {total_params}")

    optimizer = AdamW(model.parameters(), lr=params.get('lr', 1e-3), weight_decay=params.get('weight_decay', 1e-4))
    epochs = params.get('epochs', 10)
    best_val_score = 0.0
    best_epoch = 0
    history = {'train_loss': [], 'val_loss': [], 'val_intent_acc': [], 'val_slot_f1': [], 'val_combined': []}
    
    print("\nStarting Baseline Training...")
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        model.train()
        total_train_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        for batch in train_pbar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            seq_lens = batch['seq_lens'].to(device)
            intent_labels = batch['intent_label'].to(device)
            slot_labels = batch['slot_labels'].to(device)
            
            loss, _, _ = model(
                input_ids, 
                attention_mask=attention_mask, 
                seq_lens=seq_lens,
                intent_labels=intent_labels,
                slot_labels=slot_labels
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_train_loss = total_train_loss / len(train_loader)
        
        # 3. Validation & Tracking
        val_metrics = evaluate_model(model, dev_loader, device, id2slot)
        
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in dev_loader:
                loss, _, _ = model(
                    batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device),
                    seq_lens=batch['seq_lens'].to(device),
                    intent_labels=batch['intent_label'].to(device),
                    slot_labels=batch['slot_labels'].to(device)
                )
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(dev_loader)
        
        elapsed_time = time.time() - start_time
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_intent_acc'].append(val_metrics['intent_accuracy'])
        history['val_slot_f1'].append(val_metrics['slot_micro_f1'])
        history['val_combined'].append(val_metrics['combined_score'])
        
        print(f"Epoch {epoch} | Time: {elapsed_time:.2f}s | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Val Intent Acc: {val_metrics['intent_accuracy']:.2f}% | Val Slot F1: {val_metrics['slot_micro_f1']:.2f}% | Combined: {val_metrics['combined_score']:.2f}%")
        
        # 4. Lean Checkpointing
        if val_metrics['combined_score'] > best_val_score:
            best_val_score = val_metrics['combined_score']
            best_epoch = epoch
            checkpoint_path = 'experiments/baseline/best_checkpoint.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'validation_score': best_val_score
            }, checkpoint_path)
            print(f"*** Peak Validation Score! Checkpoint saved to {checkpoint_path} ***")

    # 5. Artifact Generation
    with open('experiments/baseline/metrics_history.json', 'w') as f:
        json.dump(history, f, indent=4)
        
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), history['train_loss'], label='Train Loss')
    plt.plot(range(1, epochs + 1), history['val_loss'], label='Val Loss')
    plt.title('Loss Curves')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), history['val_intent_acc'], label='Intent Acc')
    plt.plot(range(1, epochs + 1), history['val_slot_f1'], label='Slot F1')
    plt.title('Validation Metrics (%)')
    plt.legend()
    plt.savefig('experiments/baseline/learning_curves.png')
    
    print("\n" + "="*50)
    print("TRAINING COMPLETED")
    print(f"Optimal Epoch Index: {best_epoch}")
    print(f"Highest Validation Score (Combined): {best_val_score:.2f}%")
    print(f"Saved Checkpoint Filename: {checkpoint_path}")
    print("="*50)

if __name__ == '__main__':
    train()