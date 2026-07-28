import torch

def train_epoch(model, loss_func, optimizer, device, data_loader):
    """
    Trains single epoch
    """
    # transfer model to available device like cuda or cpu or mps
    model.to(device)
    # set the model to training mode
    model.train()
    total_loss = 0
    correct_prediction = 0
    total_prediction = 0
    for batch_index, (feature, target) in enumerate(data_loader):
        feature = feature.to(device)
        target = target.to(device)
        # reset the gradient
        optimizer.zero_grad()
        # forward pass
        output = model(feature)
        # initialize the loss
        loss = loss_func(output,target)
        # back prop
        loss.backward()
        # update the parameter 
        optimizer.step()

        # add loss of each batch
        total_loss += loss.item()

        _,predicted_index = output.max(1)
        correct_prediction += predicted_index.eq(target).sum().item()

        total_prediction += target.size(0)

    accuracy = (correct_prediction/total_prediction) * 100

    return model, accuracy, total_loss

def evaluate(model,device,data_loader):
    # set to evaluation mode
    model.eval()
    total_prediction = 0
    total_correct_prediction = 0

    with torch.no_grad():
        for idx, (feature, target) in enumerate(data_loader):
            feature = feature.to(device)
            target = target.to(device)

            output = model(feature)

            _, predicted_idx= output.max(1)
            total_correct_prediction += predicted_idx.eq(target).sum().item()
            total_prediction += target.size(0)

    acc = (total_correct_prediction/total_prediction)*100

    # reset back to training mode
    model.train()
    return acc

def train(model, loss_func, optimizer, device, train_data_loader, test_data_loader, epochs):
    for epoch in range(epochs):
        my_model, acc, loss = train_epoch(model, loss_func, optimizer, device, train_data_loader)
        acc = evaluate(my_model, device, test_data_loader)

        print(f"epochs: {epoch} -> acc: {acc} -> loss: {loss}")