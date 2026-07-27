import torch

def main():
    pass 

def Evaluate(model, test_loader, device):
    model.eval()
    model_acc = 0
    with torch.no_grad():
        for batch_idx, (feature, target) in enumerate(test_loader):
            feature, target = feature.to(device), target.to(device)
            output = model(feature)

    return model_acc

if __name__ == "__main__":
    main()