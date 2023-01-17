import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
from Custom_dataloader.configs import*

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_predictions(model, loader, output_csv='submission.csv'):
    preds = []
    filenames = []
    model.eval()

    for x, y, files in tqdm(loader):
        x = x.to(DEVICE)
        with torch.no_grad():
            pred = model(x).argmax(1)
            preds.append(pred.cpu().numpy())
            filenames += files
    df = pd.DataFrame({'image': filenames, 'level':np.concatenate(preds, axis=0)})
    df.to_csv(output_csv, index=False)
    model.train()
    print('Done with predictions')

def check_accuracy(loader, model, device='cuda'):
    model.eval()
    all_preds= []
    all_labels = []
    num_corrects = 0
    num_samples = 0

    for x, y, filename in tqdm(loader):
      x = x.to(device)
      y = y.to(device)  
      with torch.no_grad():
        scores = model(x)   
      _, predictions = scores.max(1) 
      num_corrects += (predictions == y).sum()
      num_samples += predictions.shape[0]   
      # add to list
      all_preds.append(predictions.detach().cpu().numpy())
      all_labels.append(y.detach().cpu().numpy())
    print(
      f'Got {num_corrects} / {num_samples} with accuracy {float(num_corrects) / float(num_samples)*100:.2f}'
  )
    model.train()
    return np.concatenate(all_preds, axis=0, dtype=np.int64),\
    np.concatenate(all_labels, axis=0, dtype=np.int64)

def save_checkpoint(state, filename='my_checkpoint.pth.tar'):
    print('saving checkpoint')
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer, lr):
    print('loading checkpoint')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    # If we dont do this then the optimizer will have the lr of old checkpoints
    for param_group in optimizer.param_groups:
      param_group['lr'] = lr


def train_one_epoch(loader, model, optimizer, loss_fn, scaler, device):
    losses = []
    loop = tqdm(loader)

    for batch_id, (data, targets, _) in enumerate(loop):
      data = data.to(device)
      targets = targets.to(device)

      with torch.cuda.amp.autocast():
        scores = model(data)
        loss = loss_fn(scores, targets)

      losses.append(loss.item())

      optimizer.zero_grad()
      scaler.scale(loss).backward()
      scaler.step(optimizer)
      scaler.update()
      loop.set_postfix(loss=loss.item())

    print(f'Loss average over epoch: {sum(losses) / len(losses)}')

#  Train loop
