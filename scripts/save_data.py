from torchvision import datasets, transforms


def main():
    trainset = datasets.CIFAR100(root="./data", train=True, download=True)
    testset = datasets.CIFAR100(root="./data", train=False, download=True)

    print(len(trainset))
    print(len(testset))


if __name__ == "__main__":
    main()
