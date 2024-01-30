import torch
from torchvision import transforms
from PIL import Image
from ResNet18 import ResNet18

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model and set it to evaluation mode
model = ResNet18(dropout_rate=0.2).to(device)
model.load_state_dict(torch.load('model.pkl', map_location=device))
model.eval()

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
])

image_path = ""

while True:
    # Input path to the picture from the user
    image_path = input("Enter the path to the image: ")

    if image_path == "stop":
        break

    try:
        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)

        # Make a prediction using the model
        with torch.no_grad():
            output = model(image)

        predicted_value = output.item()

        if predicted_value > 0.5:
            probability = (predicted_value) * 100
            print(f"Male ({probability:.4f} %)")
        else:
            probability = (1 - predicted_value) * 100
            print(f"Female ({probability:.4f} %)")

    except FileNotFoundError:
        print("File not found. Please provide a valid image path.")
    except Exception as e:
        print(f"An error occurred: {e}")
