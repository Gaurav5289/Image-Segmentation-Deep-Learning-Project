import torch
from tqdm import tqdm
from collections import OrderedDict

# --- Imports needed by THIS function ---
from ML_Pipeline.utils import AverageMeter, iou_score

# --- REMOVED unused imports like os, yaml, pandas, albumentations, UNetPP, DataSet, etc. ---


# ====================== VALIDATION FUNCTION ======================
# --- Added 'device' as an argument ---
def validate(deep_sup, val_loader, model, criterion, device):
    """
    Validation loop.
    """
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}

    # switch to evaluate mode
    model.eval()
    
    # --- REMOVED hard-coded device. It's now passed in. ---
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader), desc="Validation", leave=False)
        for input, target, _ in val_loader:
            input = input.to(device)
            target = target.to(device)

            # compute output
            if deep_sup:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou = iou_score(outputs[-1], target)
            else:
                output = model(input)
                loss = criterion(output, target)
                iou = iou_score(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))

            postfix = OrderedDict([
                ('val_loss', avg_meters['loss'].avg), # <-- Changed 'loss' to 'val_loss' for clarity
                ('val_iou', avg_meters['iou'].avg),   # <-- Changed 'iou' to 'val_iou' for clarity
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])