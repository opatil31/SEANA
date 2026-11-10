"""Download and verify both CIFAR-10 splits before submitting a run grid."""

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/cifar10")
    args = parser.parse_args()
    from torchvision.datasets import CIFAR10

    for train in (True, False):
        dataset = CIFAR10(args.data_root, train=train, download=True)
        print(f"{'train' if train else 'test'}: {len(dataset)} images")


if __name__ == "__main__":
    main()
