from evaluate import Evaluate
def main():
    pass 

def train_epoch(model, loss_function, optimizer, train_loader, device):
    # allocate model to device memory 

    model = model.to(device)
    model.train()

    epoch_loss = 0
    total_batch = len(train_loader)

    for batch_idx, (feature, target) in enumerate(train_loader):
        # reset the existing gradient 
        optimizer.zero_grad()

        # allocate feature and target to device memory
        feature = feature.to(device)
        target = target.to(device)

        # build a forward pass
        output = model(feature)
        # initialize the loss 
        loss = loss_function(output, target)
        # perform packprop
        loss.backward()
        # update the paramter
        optimizer.step()

        # update the loss 
        loss_value  = loss.item()

        # since the loss_value is loss of single batch we will add batch loss to epoch loss to get the total loss of epoch
        epoch_loss += loss_value


    avg_loss = epoch_loss/total_batch
    return model, avg_loss


def Train(model, loss_func, optimizer, epochs, train_loader, test_loader, device):

    for epoch in range(epochs):
        models, avg_loss = train_epoch(model, loss_func, optimizer, train_loader, device)
        model_accuracy = Evaluate(models, test_loader, device)

        print(f"Epoch:{epoch} -> loss:{avg_loss} -> model accuracy:{model_accuracy}")

if __name__ == "__main__":
    main()