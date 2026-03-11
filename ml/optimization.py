import torch
import torch.nn as nn
from torch.optim import RMSprop
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device} device")

def train(model, dataloader, learning_rate = 1e-3, batch_size = 8, 
          loss_fn = nn.CrossEntropyLoss(), optimizer = RMSprop()):
    size = len(dataloader.dataset)
    optimizer = optimizer(model.parameters(), lr = learning_rate)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    return optimizer.state_dict()

def test(model, dataloader, loss_fn = nn.CrossEntropyLoss()):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return test_loss

def optimizer_loop(model, train_loader, val_loader, save_path, epochs = 50):
    best_val_loss = 0
    file_path = Path(save_path)
    loss_log = f"{file_path.stem}.txt"
    for i in range(epochs):
        print(f"----------Epoch {i + 1}----------")
        opt_state_dict = train(model, train_loader)
        val_loss = test(model, val_loader)
        with open(loss_log, "a") as f:
            f.write(f"{i + 1}, {val_loss}\n")
        if i == 0:
            best_val_loss = val_loss
        elif val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch" : i + 1,
                "model_state_dict" : model.state_dict(),
                "optimizer_state_dict" : opt_state_dict,
                "loss" : best_val_loss
            }, save_path)
            print(f"Best Validation Loss Updated: {best_val_loss}")
    print(f"Finished! - Best Validation Loss: {best_val_loss}")