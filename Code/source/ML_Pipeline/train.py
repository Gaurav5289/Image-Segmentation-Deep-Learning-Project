import torch
from tqdm import tqdm
from collections import OrderedDict

# --- Imports needed by THESE functions ---
from ML_Pipeline.utils import AverageMeter, iou_score

# ====================== TRAIN FUNCTION ======================
def train(deep_sup, train_loader, model, criterion, optimizer, device):
    """
    Training loop for one epoch.
    """
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}

    model.train()
    pbar = tqdm(total=len(train_loader), desc="Training", leave=False)

    for inputs, targets, _ in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        # ----------------------------------------------------
        # --- 🐛 HOTFIX 1 (Shape): Fix shape mismatch ---
        if targets.shape[-1] == 1:
            targets = targets.permute(0, 3, 1, 2)
        
        # --- 🚀 NEW HOTFIX (Type): Convert mask to float ---
        targets = targets.float()
        # ----------------------------------------------------

        # Forward pass
        if deep_sup:
            outputs = model(inputs)
            loss = sum(criterion(output, targets) for output in outputs) / len(outputs)
            iou = iou_score(outputs[-1], targets)
        else:
            output = model(inputs)
            loss = criterion(output, targets)
            iou = iou_score(output, targets)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track metrics
        avg_meters['loss'].update(loss.item(), inputs.size(0))
        avg_meters['iou'].update(iou, inputs.size(0))

        pbar.set_postfix(OrderedDict([
            ('loss', f"{avg_meters['loss'].avg:.4f}"),
            ('iou', f"{avg_meters['iou'].avg:.4f}")
        ]))
        pbar.update(1)

    pbar.close()
    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])


# ====================== VALIDATION FUNCTION ======================
def validate(deep_sup, val_loader, model, criterion, device):
    """
    Validation loop.
    """
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}

    model.eval()
    with torch.no_grad():
        pbar = tqdm(total=len(val_loader), desc="Validation", leave=False)
        for input, target, _ in val_loader:
            input = input.to(device)
            target = target.to(device)

            # ----------------------------------------------------
            # --- 🐛 HOTFIX 1 (Shape): Fix shape mismatch ---
            if target.shape[-1] == 1:
                target = target.permute(0, 3, 1, 2)
            
            # --- 🚀 NEW HOTFIX (Type): Convert mask to float ---
            target = target.float()
            # ----------------------------------------------------

            # compute output
            if deep_sup:
                outputs = model(input)
                loss = sum(criterion(output, target) for output in outputs) / len(outputs)
                iou = iou_score(outputs[-1], target)
            else:
                output = model(input)
                loss = criterion(output, target)
                iou = iou_score(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))

            pbar.set_postfix(OrderedDict([
                ('val_loss', f"{avg_meters['loss'].avg:.4f}"),
                ('val_iou', f"{avg_meters['iou'].avg:.4f}")
            ]))
            pbar.update(1)

        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])