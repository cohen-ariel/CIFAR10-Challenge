import torch
from train_model_q1 import myCNN, get_cifar10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate_model_q1():
    # load data
    _, test_loader = get_cifar10()

    # initialize model and load weights
    model = myCNN().to(device)
    model.load_state_dict(torch.load('model_q1.pkl', map_location=lambda storage, loc: storage))

    # find accuracy on the test set
    model.eval()

    correct = 0
    total = 0
    for images, labels in test_loader:
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    test_accuracy = 100 * correct / total
    print(f"Test Accuracy of the model on the 10000 test images: {round(test_accuracy, 4)}%")

    model.train()


def main():
    evaluate_model_q1()


if __name__ == "__main__":
    main()
