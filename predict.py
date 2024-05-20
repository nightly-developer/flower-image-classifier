import argparse
import torch
import torch.nn as nn
from torchvision import transforms, models
import json
from PIL import Image
import torch.nn.functional as F

# Function to define the classifier based on the architecture
def get_classifier(arch, HU=4096):
    if arch == 'vgg':
        classifier = nn.Sequential(
            nn.Linear(25088, HU),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(HU, 102),
            nn.LogSoftmax(dim=1)
        )
    elif arch == 'resnet':
        classifier = nn.Sequential(
            nn.Linear(2048, HU),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(HU, 102)
        )
    elif arch == 'densenet':
        classifier = nn.Sequential(
            nn.Linear(1024, HU),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(HU, 102)
        )
    return classifier

# Function to process an image for model input
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns a PyTorch tensor
    '''
    # Load the image using PIL
    pil_image = Image.open(image_path)

    # Define the image transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Apply the transforms to the image
    pil_image = transform(pil_image)

    return pil_image

# Function to predict the class of an image using a trained model
def predict(model, image_path, device, topk):
    # Load and process the image
    processed_image = process_image(image_path).unsqueeze(0)

    # Move the image tensor to the same device as the model (CPU or GPU)
    model.to(device)
    processed_image = processed_image.to(device)
    
    # Set the model to evaluation mode
    model.eval()

    with torch.no_grad():
        output = model(processed_image)
        probabilities = F.softmax(output, dim=1)
        # Criteria: Top K classes
        top_probs, top_indices = probabilities.topk(topk)

    # Convert the probabilities and indices to numpy arrays
    top_probs = top_probs.cpu().numpy().squeeze()
    top_indices = top_indices.cpu().numpy().squeeze()

    return top_probs, top_indices

# wrapper function to redefine class_to_idx
def find_key_by_value(value):
    class_to_idx = {'1': 0, '10': 1, '100': 2, '101': 3, '102': 4, '11': 5, '12': 6, '13': 7, '14': 8, '15': 9, '16': 10, '17': 11, '18': 12, '19': 13, '2': 14, '20': 15, '21': 16, '22': 17, '23': 18, '24': 19, '25': 20, '26': 21, '27': 22, '28': 23, '29': 24, '3': 25, '30': 26, '31': 27, '32': 28, '33': 29, '34': 30, '35': 31, '36': 32, '37': 33, '38': 34, '39': 35, '4': 36, '40': 37, '41': 38, '42': 39, '43': 40, '44': 41, '45': 42, '46': 43, '47': 44, '48': 45, '49': 46, '5': 47, '50': 48, '51': 49, '52': 50,'53': 51, '54': 52, '55': 53, '56': 54, '57': 55, '58': 56, '59': 57, '6': 58, '60': 59, '61': 60, '62': 61, '63': 62, '64': 63, '65': 64, '66': 65, '67': 66, '68': 67, '69': 68, '7': 69, '70': 70, '71': 71, '72': 72, '73': 73, '74': 74, '75': 75, '76': 76, '77': 77, '78': 78, '79': 79, '8': 80, '80': 81, '81': 82, '82': 83, '83': 84, '84': 85, '85': 86, '86': 87, '87': 88, '88': 89, '89': 90, '9': 91, '90': 92, '91': 93, '92': 94, '93': 95, '94': 96, '95': 97, '96': 98, '97': 99, '98': 100, '99': 101}
    for key, val in class_to_idx.items():
        if val == value:
            return key

def main():
    # Define command line arguments
    parser = argparse.ArgumentParser(description="Predict flower name from an image")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    parser.add_argument("checkpoint", type=str, help="Path to the checkpoint file")
    parser.add_argument("--top_k", type=int, default=1, help="Return top K most likely classes")
    parser.add_argument("--category_names", type=str, default="cat_to_name.json", help="Path to category names JSON file")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for inference")
    args = parser.parse_args()
   
    # Setting parameters
    checkpoint_path = args.checkpoint
    image_path = args.image_path
    cat_file = args.category_names if args is not None else 'cat_to_name.json' 
    
    # Criteria: Predicting with GPU
    if args.gpu and torch.cuda.is_available():
        print("Using GPU")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Loading the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    arch = checkpoint['model_architecture']

    # Instantiate an empty model and load the classifier
    if arch == 'vgg':
        loaded_model = models.vgg19(weights=None)
        loaded_model.classifier = get_classifier(arch)
    elif arch == 'resnet':
        loaded_model = models.resnet152(weights=None)
        loaded_model.fc = get_classifier(arch)
    elif arch == 'densenet':
        loaded_model = models.densenet121(weights=None)
        loaded_model.classifier = get_classifier(arch)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")
    
    # Loading the state dictionary
    loaded_model.load_state_dict(checkpoint['model_state_dict'])

    # Criteria: Displaying class names
    with open(cat_file, 'r') as f:
        cat_to_name = json.load(f)
        
    # Criteria: Predicting classes
    top_probs, top_indices = predict(loaded_model, image_path, device, args.top_k)

    # Top k indices
    if args.top_k > 1:
        for (prob, idx) in zip(top_probs, top_indices):
            actual_idx = find_key_by_value(idx)
            flower_name = cat_to_name[actual_idx]
            print(f"Predicted index {idx} Actual index: {actual_idx} Class: {flower_name} - Probability: {prob:.3f}")
    else:
        actual_idx = find_key_by_value(top_indices)
        flower_name = cat_to_name[actual_idx]
        print(f"Predicted index {top_indices} Actual index: {actual_idx} Class: {flower_name} - Probability: {top_probs:.3f}")

if __name__ == "__main__":
    main()