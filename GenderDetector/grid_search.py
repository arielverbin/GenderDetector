import os
from train import Trainer
from dataset import init_datasets


# Create a function to train and evaluate a model for a given set of hyperparameters
def train_evaluate_model(learning_rate, batch_size, weight_decay, dropout_rate):
    # Train the model with the current hyperparameters
    train_dataset, validation_dataset, _ = init_datasets()
    trainer = Trainer(train_dataset, validation_dataset)
    trainer.define_model(dropout_rate=dropout_rate, learning_rate=learning_rate, weight_decay=weight_decay,
                         batch_size=batch_size)

    trainer.train(num_epochs=30)

    _, _, validation_f1, _, _ = trainer.evaluate()

    print(f"==> Current training resulted with F1={validation_f1}")
    # Return the F1 score
    return validation_f1

# Define hyperparameter ranges
learning_rates = [0.001, 0.005, 0.01]
batch_sizes = [32, 64]
weight_decays = [1e-5, 1e-4]
dropout_rates = [0.2, 0.3]

best_f1_score = 0.0
best_hyperparameters = {}


# Create the directory for saving model checkpoints if it doesn't exist
checkpoint_dir = 'trained_models'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# Iterate over hyperparameters
for learning_rate in learning_rates:
    for batch_size in batch_sizes:
        for weight_decay in weight_decays:
            for dropout_rate in dropout_rates:

                # The Google colab stops running for some reason :(. Skipping the combinations already checked.
                if (learning_rate, batch_size, weight_decay, dropout_rate) in already_checked:
                    continue

                current_f1 = train_evaluate_model(learning_rate, batch_size, weight_decay, dropout_rate)

                # Check if the current combination of hyperparameters resulted in a better F1 score
                if current_f1 > best_f1_score:
                    best_f1_score = current_f1
                    best_hyperparameters = {
                        'learning_rate': learning_rate,
                        'batch_size': batch_size,
                        'weight_decay': weight_decay,
                        'dropout_rate': dropout_rate
                    }


# Print the best hyperparameters and corresponding F1 score
print("Best Hyperparameters:", best_hyperparameters)
print("Best F1 Score:", best_f1_score)
