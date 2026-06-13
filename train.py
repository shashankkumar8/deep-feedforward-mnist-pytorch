"""
Project: Deep Feedforward Neural Network 
         for Handwritten Digit Recognition
Author: Shashank Kumar
GitHub: github.com/shashankkumar8
Key Concepts: Neural Networks, Backpropagation,
              Gradient Descent, Linear Algebra,
              Softmax, Cross-Entropy, Regularization
Result: 98%+ test accuracy on MNIST
"""

import os
import time
import random
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# Set a fixed seed for reproducible training and evaluation behavior.
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Use GPU when available, otherwise fall back to CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FeedforwardMNIST(nn.Module):
    """Deep feedforward neural network for MNIST digit classification.

    The network uses fully connected layers. Each hidden layer applies
    a linear transformation y = Wx + b, followed by nonlinearity and
    regularization.
    """

    def __init__(self):
        super().__init__()

        # The first layer takes the flattened 28x28 image as a 784-vector input.
        # Hidden layers learn feature representations with decreasing width.
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the network.

        The forward pass is matrix multiplication followed by bias addition
        in each linear layer. This builds the logits used for prediction.
        """
        return self.model(x)


def load_data(batch_size: int = 64):
    """Load the MNIST dataset and create training, validation, and test loaders."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_dataset = datasets.MNIST(
        root="data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="data", train=False, download=True, transform=transform
    )

    train_size = int(len(train_dataset) * 0.8)
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    return train_loader, val_loader, test_loader, test_dataset


def compute_accuracy_and_loss(model, data_loader, criterion):
    """Compute loss and accuracy for a dataset under evaluation mode."""
    model.eval()
    cumulative_loss = 0.0
    cumulative_correct = 0
    cumulative_samples = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            cumulative_loss += loss.item() * images.size(0)
            predictions = logits.argmax(dim=1)
            cumulative_correct += (predictions == labels).sum().item()
            cumulative_samples += labels.size(0)

    return cumulative_loss / cumulative_samples, cumulative_correct / cumulative_samples


def train(model, train_loader, val_loader, criterion, optimizer, epochs: int):
    """Train the model and save the best-performing validation checkpoint."""
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "epoch_time": [],
    }
    best_val_accuracy = 0.0
    best_model_path = "mnist_model.pth"

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start = time.time()

        running_loss = 0.0
        running_correct = 0
        running_samples = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(images)

            # CrossEntropyLoss combines: softmax(logits) + negative log likelihood.
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            # Accumulate training statistics for this epoch.
            running_loss += loss.item() * images.size(0)
            predictions = logits.argmax(dim=1)
            running_correct += (predictions == labels).sum().item()
            running_samples += labels.size(0)

        epoch_train_loss = running_loss / running_samples
        epoch_train_accuracy = running_correct / running_samples

        epoch_val_loss, epoch_val_accuracy = compute_accuracy_and_loss(
            model, val_loader, criterion
        )

        epoch_duration = time.time() - epoch_start

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_accuracy"].append(epoch_train_accuracy)
        history["val_accuracy"].append(epoch_val_accuracy)
        history["epoch_time"].append(epoch_duration)

        # Save the model whenever validation accuracy improves.
        if epoch_val_accuracy > best_val_accuracy:
            best_val_accuracy = epoch_val_accuracy
            torch.save(model.state_dict(), best_model_path)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_accuracy:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_accuracy:.4f} | "
            f"Time: {epoch_duration:.2f}s"
        )

    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    return history, best_model_path


def evaluate_on_test(model, test_loader, criterion):
    """Evaluate the fitted model on the test dataset and compute per-class metrics."""
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_samples = 0

    num_classes = 10
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    class_correct = np.zeros(num_classes, dtype=int)
    class_counts = np.zeros(num_classes, dtype=int)

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            test_loss += loss.item() * images.size(0)
            probabilities = torch.softmax(logits, dim=1)
            predictions = probabilities.argmax(dim=1)

            test_correct += (predictions == labels).sum().item()
            test_samples += labels.size(0)

            for target, prediction in zip(labels.cpu().numpy(), predictions.cpu().numpy()):
                confusion_matrix[target, prediction] += 1
                class_counts[target] += 1
                if target == prediction:
                    class_correct[target] += 1

    avg_test_loss = test_loss / test_samples
    test_accuracy = test_correct / test_samples
    per_class_accuracy = class_correct / class_counts.astype(float)

    return avg_test_loss, test_accuracy, confusion_matrix, per_class_accuracy


def plot_training_history(history):
    """Save a 2x2 figure containing training and validation curves."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].plot(history["train_loss"], marker="o", label="Train Loss")
    axes[0, 0].set_title("Training Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True)

    axes[0, 1].plot(history["val_loss"], marker="o", color="orange", label="Validation Loss")
    axes[0, 1].set_title("Validation Loss")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].grid(True)

    axes[1, 0].plot(history["train_accuracy"], marker="o", color="green", label="Train Accuracy")
    axes[1, 0].set_title("Training Accuracy")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].grid(True)
    axes[1, 0].set_ylim(0, 1)

    axes[1, 1].plot(history["val_accuracy"], marker="o", color="red", label="Validation Accuracy")
    axes[1, 1].set_title("Validation Accuracy")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Accuracy")
    axes[1, 1].grid(True)
    axes[1, 1].set_ylim(0, 1)

    fig.suptitle("MNIST Training and Validation Curves", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig("training_curves.png", dpi=150)
    plt.close(fig)


def plot_confusion_matrix(confusion_matrix):
    """Save a confusion matrix image for the test set."""
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(confusion_matrix, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    classes = [str(i) for i in range(10)]
    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True Label",
        xlabel="Predicted Label",
        title="MNIST Confusion Matrix",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = confusion_matrix.max() / 2.0
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            ax.text(
                j,
                i,
                format(confusion_matrix[i, j], "d"),
                ha="center",
                va="center",
                color="white" if confusion_matrix[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig("confusion_matrix.png", dpi=150)
    plt.close(fig)


def unnormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalized tensor image back to an RGB-compatible numpy array."""
    mean = 0.1307
    std = 0.3081
    image = tensor.cpu().numpy().squeeze(0)
    image = image * std + mean
    image = np.clip(image, 0, 1)
    return image


def plot_sample_predictions(model, test_dataset):
    """Save a 5x5 grid image of random test predictions with confidence scores."""
    model.eval()
    indices = random.sample(range(len(test_dataset)), 25)
    fig, axes = plt.subplots(5, 5, figsize=(12, 12))

    for ax, idx in zip(axes.flatten(), indices):
        image, label = test_dataset[idx]
        image_unsqueezed = image.unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(image_unsqueezed)
            probabilities = torch.softmax(logits, dim=1)
            confidence, prediction = torch.max(probabilities, dim=1)

        image_display = unnormalize_image(image)
        ax.imshow(image_display, cmap="gray")
        ax.set_title(
            f"True: {label}\nPred: {prediction.item()} ({confidence.item() * 100:.1f}%)"
        )
        ax.axis("off")

    fig.suptitle("Sample MNIST Predictions", fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig("sample_predictions.png", dpi=150)
    plt.close(fig)


def plot_first_layer_weights(model):
    """Visualize the first layer weights as 32 learned 28x28 filter images."""
    first_linear = model.model[1]
    weights = first_linear.weight.data.cpu()
    num_filters = 32

    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    for i in range(num_filters):
        ax = axes.flat[i]
        weight_image = weights[i].reshape(28, 28).numpy()
        weight_image = (weight_image - weight_image.min()) / (
            weight_image.max() - weight_image.min() + 1e-8
        )
        ax.imshow(weight_image, cmap="viridis")
        ax.set_title(f"Filter {i}")
        ax.axis("off")

    fig.suptitle("First Layer Weight Visualization", fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig("weight_visualization.png", dpi=150)
    plt.close(fig)


def count_parameters(model):
    """Return the total number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_summary(test_loss, test_accuracy, per_class_accuracy, confusion_matrix, total_parameters, history):
    """Print the final evaluation results and training summary."""
    print("\nFinal evaluation results")
    print("------------------------")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Total Parameters: {total_parameters:,}")
    print(f"Average Epoch Time: {np.mean(history['epoch_time']):.2f}s")

    print("\nPer-class accuracy:")
    for digit, acc in enumerate(per_class_accuracy):
        print(f"Digit {digit}: {acc * 100:.2f}%")

    print("\nConfusion matrix summary:")
    print(confusion_matrix)
    print("\nMost common errors by true digit:")
    for true_digit in range(confusion_matrix.shape[0]):
        row = confusion_matrix[true_digit].copy()
        row[true_digit] = 0
        predicted = row.argmax()
        if row.max() > 0:
            print(
                f"  True {true_digit} most often confused with {predicted} "
                f"({row[predicted]} times)"
            )
        else:
            print(f"  True {true_digit} has no confusions.")


def main():
    """Main entry point for training, evaluation, plotting, and model persistence."""
    os.makedirs("data", exist_ok=True)

    train_loader, val_loader, test_loader, test_dataset = load_data(batch_size=64)
    model = FeedforwardMNIST().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    history, best_model_path = train(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        epochs=10,
    )

    # Load the best model as determined by validation accuracy.
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    test_loss, test_accuracy, confusion_matrix, per_class_accuracy = evaluate_on_test(
        model, test_loader, criterion
    )

    plot_training_history(history)
    plot_confusion_matrix(confusion_matrix)
    plot_sample_predictions(model, test_dataset)
    plot_first_layer_weights(model)

    total_parameters = count_parameters(model)
    print_summary(
        test_loss,
        test_accuracy,
        per_class_accuracy,
        confusion_matrix,
        total_parameters,
        history,
    )


if __name__ == "__main__":
    main()
