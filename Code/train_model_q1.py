import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# Constants
PLOT_FREQ = 5

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 100
batch_size = 128
learning_rate = 0.001


def get_cifar10():
    transform = transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.ColorJitter(brightness=0.1),
         transforms.RandomHorizontalFlip(p=0.5),
         transforms.ToTensor(),
         transforms.Normalize((0.491, 0.482, 0.447), (0.247, 0.243, 0.262))])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                 download=True, transform=transform)

    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                              shuffle=False)

    return train_loader, test_loader


def plot_loss(train_loss_lst, test_loss_lst):
    """Gets train and test loss lists, and plots loss values
    for every epoch"""
    plt.style.use("ggplot")
    plt.plot([i * PLOT_FREQ for i in range(len(train_loss_lst))], train_loss_lst,
             label="train loss")
    plt.plot([i * PLOT_FREQ for i in range(len(test_loss_lst))], test_loss_lst,
             label="test loss")

    plt.title("Loss Value for Every Epoch")
    plt.xlabel("epochs")
    plt.ylabel("loss value")
    plt.legend()

    plt.show()


def plot_error(train_error_lst, test_error_lst):
    """Gets train and test error lists, and plots error values
    for every epoch"""
    plt.style.use("ggplot")
    plt.plot([i for i in range(len(train_error_lst))], train_error_lst,
             label="train error")
    plt.plot([i for i in range(len(test_error_lst))], test_error_lst,
             label="test error")

    plt.title("Error Value for Every Epoch")
    plt.xlabel("epochs")
    plt.ylabel("error value")
    plt.legend()

    plt.show()


class myCNN(nn.Module):
    def __init__(self):
        super(myCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
        )

        self.fc = nn.Linear(8 * 8 * 48, 10)
        self.dropout = nn.Dropout(p=0.5)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)

        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)
        return self.logsoftmax(out)


def train_model_q1(model, train_loader, test_loader):
    # initialize
    n_total_steps = len(train_loader)
    train_loss_lst, test_loss_lst = [], []
    train_error_lst, test_error_lst = [], []

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, num_epochs+1):
        correct = 0
        total = 0

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            train_loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            # Accumulate number of correct predictions and total predictions for each train batch
            model.eval()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            model.train()

            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {train_loss.item():.4f}')

        # calculate loss only every PLOT_FREQ'th epoch
        model.eval()

        if epoch == 1 or epoch % PLOT_FREQ == 0:

            # get train loss
            train_loss_lst.append(train_loss.item())

            # get test loss
            with torch.no_grad():
                # get test set
                for X2, Y2 in test_loader:
                    X_test = X2
                    Y_test = Y2

                # get predictions + loss
                test_predictions = model(X_test)

                test_loss = criterion(test_predictions, Y_test)
                test_loss_lst.append(test_loss.item())

        # get train error
        train_error = 1 - (correct / total)
        train_error_lst.append(train_error)

        # get test error
        total = 0
        correct = 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_error = 1 - (correct / total)
        test_error_lst.append(test_error)

        model.train()

        if test_error < 0.2:
            print(f"Stopped at epoch: {epoch}")
            break

    print('Finished Training')

    # save weights
    path = 'model_q1.pkl'
    torch.save(model.state_dict(), path)

    # display final accuracy
    print()
    print('Number of parameters: ', sum(param.numel() for param in model.parameters()))
    print(f"Test Accuracy of the model on the 10000 test images: {round(100*(1 - test_error), 4)}%")

    # display convergence graphs
    plot_loss(train_loss_lst, test_loss_lst)
    plot_error(train_error_lst, test_error_lst)


def main():
    # load data
    train_loader, test_loader = get_cifar10()

    # initialize model
    model = myCNN().to(device)
    print('Number of parameters: ', sum(param.numel() for param in model.parameters()))

    train_model_q1(model, train_loader, test_loader)


if __name__ == "__main__":
    main()
