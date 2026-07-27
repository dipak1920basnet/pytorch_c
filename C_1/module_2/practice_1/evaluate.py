import torch

def main():
    pass 

def Evaluate(model, test_loader, device):
    model.eval()
    total_prediction = 0
    correct_prediction = 0
    with torch.no_grad():
        for batch_idx, (feature, target) in enumerate(test_loader):
            feature, target = feature.to(device), target.to(device)
            output = model(feature)
            total_prediction += target.size(0)

            # check the number of correct prediction in a batch 
            _, predicted_ind = output.max(1)
            correct_prediction += predicted_ind.eq(target).sum().item()

    model_acc = (correct_prediction/total_prediction) * 100
    return model_acc

if __name__ == "__main__":
    main()