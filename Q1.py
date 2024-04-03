from torchvision import datasets, transforms
import torch
from torch import nn
from torch.utils.data import DataLoader
from model import LeNet
from torchinfo import summary
from torchmetrics import Accuracy
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import argparse

torch.manual_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # For M1 chip, torch.device("mps") is used instead of torch.device("cuda") for gpu training.
print(DEVICE)

def preprocess(BATCH_SIZE=32, DEVICE=DEVICE):
    # Download the MNIST dataset
    
    normalize = transforms.Normalize((0.1307,), (0.3081,))
    mnist_transforms  = transforms.Compose([transforms.ToTensor(), normalize])  

    train_val_dataset = datasets.MNIST(root="./datasets/", train=True, download=True, transform=mnist_transforms)
    test_dataset = datasets.MNIST(root="./datasets/", train=False, download=True, transform=mnist_transforms)

    # Split the dataset into train and validation
    train_size = int(0.8 * len(train_val_dataset))
    val_size = len(train_val_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset=train_val_dataset, lengths=[train_size, val_size])

    # Create dataloaders
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    return train_dataloader, val_dataloader, test_dataloader

def train(model, train_dataloader, val_dataloader, num_epochs, learning_rate, weight_decay, MODEL_SAVE_PATH, DEVICE):
    # Loss function, optimizer and accuracy
    model.to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    accuracy = Accuracy(task='multiclass', num_classes=10).to(DEVICE)
    
    writer = SummaryWriter()
    for epoch in tqdm(range(num_epochs)):
        # Training loop
        train_loss, train_acc = 0.0, 0.0
        for X, y in train_dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            
            model.train()
            
            y_pred = model(X)
            
            loss = loss_fn(y_pred, y)
            train_loss += loss.item()
            
            acc = accuracy(y_pred, y)
            train_acc += acc
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        train_loss /= len(train_dataloader)
        train_acc /= len(train_dataloader)
            
        # Validation loop
        val_loss, val_acc = 0.0, 0.0
        model.eval()
        with torch.inference_mode():
            for X, y in val_dataloader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                
                y_pred = model(X)
                
                loss = loss_fn(y_pred, y)
                val_loss += loss.item()
                
                acc = accuracy(y_pred, y)
                val_acc += acc
                
            val_loss /= len(val_dataloader)
            val_acc /= len(val_dataloader)
        writer.add_scalars(main_tag="Loss", tag_scalar_dict={"train/loss": train_loss, "val/loss": val_loss}, global_step=epoch)
        writer.add_scalars(main_tag="Accuracy", tag_scalar_dict={"train/acc": train_acc, "val/acc": val_acc}, global_step=epoch)  
        print(f"Epoch: {epoch}| Train loss: {train_loss: .5f}| Train acc: {train_acc: .5f}| Val loss: {val_loss: .5f}| Val acc: {val_acc: .5f}")
    
    # Saving the model
    print(f"Saving the model: {MODEL_SAVE_PATH}")
    torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH)

def test(model, test_dataloader, MODEL_SAVE_PATH, DEVICE):
    test_loss, test_acc = 0, 0
    # Loading the saved model
    print(f"Load the model: {MODEL_SAVE_PATH}")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.to(DEVICE)
    model.eval()
    # Loss function, optimizer and accuracy
    loss_fn = nn.CrossEntropyLoss()
    accuracy = Accuracy(task='multiclass', num_classes=10).to(DEVICE)
    with torch.inference_mode():
        for X, y in test_dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            y_pred = model(X)
            
            test_loss += loss_fn(y_pred, y)
            test_acc += accuracy(y_pred, y)
            
        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)

    print(f"Test loss: {test_loss: .5f}| Test acc: {test_acc: .5f}")
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='MNIST_LeNet', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mode', choices=['train', 'test'], default='train', help='choose training the neural network or testing it.')
    parser.add_argument('--print_model', default="False", help='print the model summary.')
    parser.add_argument('--hidden_channel', type=int, default=12, help='number of hidden channels in the second convolutional layer.')
    parser.add_argument('--hidden_linear', type=int, default=120, help='number of hidden neurons in the first fully connected layer.')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for training and testing.')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs for training the model.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate for training the model.')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for training the model.')
    args = parser.parse_args()

    # Download the MNIST dataset and create dataloaders
    train_dataloader, val_dataloader, test_dataloader = preprocess(args.batch_size)

    # Define the model
    modellenet = LeNet(hidden_channel=args.hidden_channel,hidden_linear=args.hidden_linear)
    MODEL_PATH = Path("models")
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    MODEL_NAME = "lenet_mnist"+"_lr_"+str(args.learning_rate)+"_wd_"+str(args.weight_decay)+"_"+str(args.hidden_channel)+"_"+str(args.hidden_linear)+".pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
    if args.print_model == "True":
        # Summary of the model
        print(summary(model=modellenet.to(DEVICE), input_size=(1, 1, 28, 28), col_width=20, col_names=['input_size', 'output_size', 'num_params', 'trainable'], row_settings=['var_names'], verbose=0))        
    if args.mode == 'train':
        # Train the model
        train(modellenet, train_dataloader, val_dataloader, args.epochs, args.learning_rate, args.weight_decay, MODEL_SAVE_PATH, DEVICE)
    elif args.mode == 'test':
        # Test the model
        test(modellenet, test_dataloader, MODEL_SAVE_PATH, DEVICE)
