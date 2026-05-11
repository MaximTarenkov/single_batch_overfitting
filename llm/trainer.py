import torch
import time
from tqdm.auto import tqdm

def train_custom(model, dataloader, optimizer, device, target_time, single_batch=None, accumulation_steps=4):
    model.train()
    model.to(device)
    losses = []
    start_time = time.time()
    iters = 0
    batch_iter = iter(dataloader)
    batch = single_batch if single_batch is not None else next(batch_iter)
    
    pbar = tqdm(total=target_time, desc="Training", unit="s", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}s [{postfix}]")
    last_elapsed = 0

    while True:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / accumulation_steps
            
        loss.backward()
        if (iters + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            
        current_loss = loss.item() * accumulation_steps
        losses.append(current_loss)
        iters += 1
        
        elapsed = time.time() - start_time
        pbar.update(elapsed - last_elapsed)
        last_elapsed = elapsed
        
        pbar.set_postfix({"loss": f"{current_loss:.4f}", "iters": iters})
        
        if elapsed >= target_time:
            break

        if single_batch is None:
            try:
                batch = next(batch_iter)
            except StopIteration:
                batch_iter = iter(dataloader)
                batch = next(batch_iter)
                
    pbar.close()
    return losses