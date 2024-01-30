import torch
from ResNet18 import ResNet18
from torch.utils.data import DataLoader
from dataset import init_datasets
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# calculate the stats given the calculated values and the true values.
def calculate_stats(outputs, true_values):
    tp = torch.sum((outputs == 1) & (true_values == 1)).item()
    fp = torch.sum((outputs == 1) & (true_values == 0)).item()
    fn = torch.sum((outputs == 0) & (true_values == 1)).item()
    tn = torch.sum((outputs == 0) & (true_values == 0)).item()

    # Calculate the evaluation metrics
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return f1, precision, accuracy, recall


if __name__ == '__main__':
    model = ResNet18(dropout_rate=0.2).to(device)

    # Load the trained model
    model.load_state_dict(torch.load('model.pkl', map_location=device))
    model.to(device)

    # Initialize the test dataset
    _, _, test_dataset = init_datasets()

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True,
                             num_workers=8, persistent_workers=True,
                             pin_memory=False, drop_last=True)

    model.eval()  # Set the model to evaluation mode

    total_outputs = torch.Tensor().to(device)  # To store all outputs
    total_labels = torch.Tensor().to(device)  # To store all true labels

    with torch.no_grad():
        for images, labels in test_loader:
            if torch.cuda.is_available():
                images = images.to(device)
                labels = labels.to(device)

            outputs = model(images)
            labels = labels.type(torch.FloatTensor).to(device)
            labels = labels.unsqueeze(1)

            # Append outputs and labels for calculating stats
            total_outputs = torch.cat((total_outputs, torch.round(outputs)), dim=0)
            total_labels = torch.cat((total_labels, labels), dim=0)

    # Calculate evaluation metrics using the calculate_stats function
    f1, precision, accuracy, recall = calculate_stats(total_outputs, total_labels)

    print(f"Test Results:")
    print(f'\tF1 Score:\t{f1:.4f}')
    print(f'\tAccuracy:\t{accuracy:.4f}')
    print(f'\tPrecision:\t{precision:.4f}')
    print(f'\tRecall:\t{recall:.4f}')

    # Display Confusion Matrix:

    # Convert predictions and labels to NumPy arrays
    predictions = total_outputs.numpy()
    labels = total_labels.numpy()

    # Calculate the confusion matrix
    cm = confusion_matrix(labels, predictions)

    # Display the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues, values_format='.0f')

    # Add labels and title
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix')

    # Show the plot
    plt.show()
