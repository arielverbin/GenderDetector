import os
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
from eval import calculate_stats
from dataset import init_datasets

# Import custom ResNet model
from ResNet18 import ResNet18

# Determine the device to be used for computations: CPU or GPU (if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Define the model and train it with the given datasets.
class Trainer:
    def __init__(self, train_dataset, val_dataset):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.model = None
        self.criterion = None

        self.dropout_rate, self.batch_size, self.lr, self.weight_decay = None, None, None, None
        self.patience = None

        self.train_loader = None
        self.val_loader = None

    def define_model(self, dropout_rate=0.3, learning_rate=0.01, weight_decay=1e-5, batch_size=64, patience=6):
        self.model = ResNet18(dropout_rate).to(device)
        self.dropout_rate = dropout_rate
        self.criterion = nn.BCELoss().to(device)

        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.patience = patience

        # Create data loaders for each dataset
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True,
                                       num_workers=8, persistent_workers=True,
                                       pin_memory=False, drop_last=True)

        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=True,
                                     num_workers=8, persistent_workers=True,
                                     pin_memory=False, drop_last=True)

    # Training the network
    def train(self, num_epochs=40):

        print("==> Now Training with:")
        print(f"\tWeight Decay:\t{self.weight_decay}\n\tBatch Size:\t{self.batch_size}\n"
              f"\tLearning Rate:\t{self.lr}\n\tDropout Rate:\t{self.dropout_rate}")

        best_val_loss = float('inf')  # Initialize the best validation loss to positive infinity
        current_patience = 0  # Initialize the patience counter

        optimizer = torch.optim.AdamW(self.model.parameters(), self.lr, weight_decay=self.weight_decay)
        # Initialize the learning rate scheduler
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5, verbose=True)

        f1, precision, accuracy, specificity = [], [], [], []

        for epoch in range(num_epochs):
            epoch_f1, epoch_precision, epoch_accuracy, epoch_specificity, epoch_loss = 0, 0, 0, 0, 0

            for i, (images, labels) in (t := tqdm(enumerate(self.train_loader), leave=True, position=0)):
                size = len(self.train_loader)
                if torch.cuda.is_available():
                    images = images.to(device)
                    labels = labels.to(device)

                # Forward
                outputs = self.model(images).to(device)
                labels = labels.type(torch.FloatTensor).to(device)
                pred = torch.round(outputs)

                # Backward
                labels = labels.unsqueeze(1)
                loss = self.criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()

                # Optimizer
                torch.nn.utils.clip_grad_norm(self.model.parameters(), max_norm=1)
                optimizer.step()

                batch_f1, batch_precision, batch_accuracy, batch_specificity = calculate_stats(pred, labels)
                epoch_f1 += (batch_f1 / size)
                epoch_precision += batch_precision / size
                epoch_accuracy += batch_accuracy / size
                epoch_specificity += batch_specificity / size

                # Accumulate batch loss
                epoch_loss += loss.item()

                t.set_postfix_str(
                    f'Epoch: {epoch + 1}/{num_epochs} Loss: {loss.item():.4f} Accuracy: {epoch_accuracy * 100:.4f}%')

            epoch_loss /= len(self.train_loader)

            # save metrics
            f1.append(torch.tensor(epoch_f1))
            precision.append(torch.tensor(epoch_precision))
            accuracy.append(torch.tensor(epoch_accuracy))
            specificity.append(torch.tensor(epoch_specificity))

            # Evaluate on the validation dataset
            val_loss, val_accuracy, val_f1, _, _ = self.evaluate()

            # Step the learning rate scheduler based on the validation loss
            scheduler.step(val_loss)

            print(f"\tLearning Rate: {optimizer.param_groups[0]['lr']:.7f}")
            print(f"\t--> Epoch {epoch + 1} Summary:\n\t\tTrain Loss:\t\t{epoch_loss:.4f}\n\t\tTrain Accuracy:\t\t"
                  f"{epoch_accuracy * 100:.4f}%\n\t\tValidation Loss:\t{val_loss:.4f}\n\t\tValidation Accuracy:"
                  f"\t{val_accuracy * 100:.4f}%\n\t\tValidation F1:\t\t{val_f1}\n")

            # Check if validation loss has improved
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                current_patience = 0  # Reset patience counter
            else:
                current_patience += 1  # Increment patience counter

            # Save model version {epoch}
            with open(f'trained_models/model_epoch{epoch + 1}.pkl', 'w'):
                torch.save(self.model.state_dict(), f'trained_models//model_epoch{epoch + 1}.pkl')

            # Check if early stopping criteria met
            if current_patience >= self.patience:
                print(f'==> Early stopping after {epoch + 1} epochs with no improvement in validation loss.')
                break

        print(f"==> Training ended.")
        return f1, precision, accuracy, specificity

    def evaluate(self):
        self.model.eval()  # Set the model to evaluation mode
        total_loss = 0

        total_outputs = torch.Tensor().to(device)  # To store all outputs
        total_labels = torch.Tensor().to(device)  # To store all true labels

        with torch.no_grad():
            for images, labels in self.val_loader:
                if torch.cuda.is_available():
                    images = images.to(device)
                    labels = labels.to(device)

                outputs = self.model(images)
                labels = labels.type(torch.FloatTensor).to(device)
                labels = labels.unsqueeze(1)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                # Append outputs and labels for calculating stats
                total_outputs = torch.cat((total_outputs, torch.round(outputs)), dim=0)
                total_labels = torch.cat((total_labels, labels), dim=0)

        self.model.train()  # Set the model back to training mode

        # Calculate evaluation metrics using the calculate_stats function
        f1, precision, accuracy, recall = calculate_stats(total_outputs, total_labels)

        # Return the average validation loss, accuracy, F1 score, precision, and specificity
        return total_loss / len(self.val_loader), accuracy, f1, precision, recall


# Will be used after finding the optimal hyperparameters.
if __name__ == '__main__':
    # Create the directory for saving model checkpoints if it doesn't exist
    checkpoint_dir = 'trained_models'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    train_dataset, validation_dataset, _ = init_datasets()

    trainer = Trainer(train_dataset, validation_dataset)
    trainer.define_model(dropout_rate=0.2, learning_rate=0.001, weight_decay=1e-5, batch_size=32, patience=20)
    trainer.train(num_epochs=200)

    val_loss, val_accuracy, val_f1, val_precision, val_specificity = trainer.evaluate()
    print(f"Train results: Val Loss: {val_loss},\n"
          f"Val Accuracy: {val_accuracy},\n"
          f"Val F1:{val_f1},\n"
          f"Val Precision: {val_precision}")
