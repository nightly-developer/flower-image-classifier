import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# Define a function for the validation pass
def validation(model, testloader, criterion, device):
    test_loss = 0
    accuracy = 0
    
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        output = model.forward(inputs)
        test_loss += criterion(output, labels).item()
        
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return test_loss, accuracy

# Define the training function
def training_model(model, epochs, device, optimizer, criterion, dataloaders):
    # Training Loop
    model.to(device)
    print("Training process initializing .....\n")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        model.eval()
        with torch.no_grad():
            valid_loss, accuracy = validation(model, dataloaders['valid'], criterion, device)

        # Criteria: Training validation log
        print("Epoch: {}/{} | ".format(epoch+1, epochs),
              "Training Loss: {:.4f} | ".format(running_loss/len(dataloaders['train'])),
              "Validation Loss: {:.4f} | ".format(valid_loss/len(dataloaders['valid'])),
              "Validation Accuracy: {:.4f}".format(accuracy/len(dataloaders['valid'])))

        running_loss = 0
        model.train()

    print("\nTraining process is now complete!!")
    return model

# Define a function to get the classifier based on the architecture
def get_classifier(arch, hidden_units):
    if arch == 'vgg':
        classifier = nn.Sequential(
            nn.Linear(25088, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_units, 102),
            nn.LogSoftmax(dim=1)
        )
    elif arch == 'resnet':
        classifier = nn.Sequential(
            nn.Linear(2048, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_units, 102)
        )
    elif arch == 'densenet':
        classifier = nn.Sequential(
            nn.Linear(1024, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_units, 102)
        )
    return classifier

# Main function
def main():
    parser = argparse.ArgumentParser(description="Train a new network on a data set")
    # Criteria: Model architecture, Model hyperparameters, Training with GPU
    parser.add_argument("data_dir", type=str, help="Path to the data directory")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--hidden_units", type=int, default=4096, help="Number of hidden units")
    parser.add_argument("--save_dir", type=str, default="", help="Directory to save checkpoints")
    parser.add_argument("--arch", type=str, choices=["vgg", "resnet", "densenet"], default="vgg", help="Architecture (vgg, resnet, densenet)")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training")
    args = parser.parse_args()

    # Extract command-line arguments
    arch = args.arch if args.arch is not None else 'vgg'
    learning_rate = args.learning_rate if args.learning_rate is not None else 0.001
    epochs = args.epochs if args.epochs is not None else 5
    hidden_units = args.hidden_units if args.hidden_units is not None else 4096
    save_dir = args.save_dir if args.save_dir is not None else "./"

    # Define data directories
    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
  
    # Define data transformations
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the datasets using ImageFolder
    image_datasets = {
        'train': datasets.ImageFolder(root=train_dir, transform=data_transforms),
        'valid': datasets.ImageFolder(root=valid_dir, transform=data_transforms),
        'test': datasets.ImageFolder(root=test_dir, transform=data_transforms)
    }

    # Define the data loaders
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=32, shuffle=True),
        'valid': DataLoader(image_datasets['valid'], batch_size=32),
        'test': DataLoader(image_datasets['test'], batch_size=32)
    }

    # Initialize the model, optimizer, and criterion based on the selected architecture
    if arch == 'vgg':
        model = models.vgg19(weights='DEFAULT')
        model.classifier = get_classifier(arch, hidden_units)
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    elif arch == 'resnet':
        model = models.resnet152(weights='DEFAULT')
        num_ftrs = model.fc.in_features
        model.fc = get_classifier(arch, hidden_units)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
    elif arch == 'densenet':
        model = models.densenet121(weights='DEFAULT')
        model.classifier = get_classifier(arch, hidden_units)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported architecture: {arch}")
    
    # Criteria: training with GPU
    if args.gpu and torch.cuda.is_available():
        print("Using GPU")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Train the model
    trained_model = training_model(model, epochs, device, optimizer, criterion, dataloaders)

    # Save the checkpoint
    checkpoint = {
        'model_state_dict': trained_model.state_dict(),
        'model_architecture': arch,
    }
    torch.save(checkpoint, f'{save_dir}{arch}.pth')

if __name__ == "__main__":
    main()
