import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

def overfit_single_batch(model, dataloader, device, iterations=200, lr=0.01):
    model.to(device)
    model.train()
    
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    data_iter = iter(dataloader)
    inputs, labels = next(data_iter)
    inputs, labels = inputs.to(device), labels.to(device)
    
    loss_history = []
    
    pbar = tqdm(range(iterations), desc="Warmup (Single Batch)")
    for i in pbar:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        curr_loss = loss.item()
        loss_history.append(curr_loss)
        if i % 10 == 0:
            pbar.set_postfix({"loss": f"{curr_loss:.4f}"})
            
    return model, loss_history

def train_model(model, trainloader, testloader, device, epochs=10, start_iteration=0, label="Model"):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    metrics = {
        'iterations': [],
        'train_loss': [],
        'test_acc': []
    }
    
    global_step = start_iteration
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(trainloader, desc=f"{label} Epoch {epoch+1}/{epochs}", leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            step_loss = loss.item()
            running_loss += step_loss
            global_step += 1
            
            pbar.set_postfix({"loss": f"{step_loss:.3f}"})
            
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        acc = 100 * correct / total
        avg_loss = running_loss / len(trainloader)
        
        metrics['iterations'].append(global_step)
        metrics['train_loss'].append(avg_loss)
        metrics['test_acc'].append(acc)
        
        print(f"[{label} Epoch {epoch+1}] Loss: {avg_loss:.4f} | Test Acc: {acc:.2f}% | Total Steps: {global_step}")
        
        scheduler.step()
        
    return metrics