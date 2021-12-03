# coding=utf-8
import torch
from datasets import get_dataset
from architectures import get_architecture
from certify import eval_certify

base_model_path = "path of base classifier"
detector_path = "path of detector"


def create_base_classifier():
    classifier = get_architecture("mnist_lenet", "mnist")
    checkpoint = torch.load(base_model_path)
    classifier.load_state_dict(checkpoint['state_dict'])
    classifier.eval()
    classifier.cuda()
    return classifier


def create_detector():
    detector = torch.nn.Sequential(
        torch.nn.Linear(10, 2048),
        torch.nn.ReLU(),
        torch.nn.Linear(2048, 2)
    )

    checkpoint = torch.load(detector_path)
    detector.load_state_dict(checkpoint['state_dict'])
    detector.eval()
    detector.cuda()
    return detector


def certify():
    dataset = get_dataset("mnist", "test")
    detector = create_detector()
    base = create_base_classifier()

    class Args:
        nb_classes = 10

        start_idx = 0
        skip = 1
        nb_max = 10000
        N0 = 300
        N = 100000
        alpha = 0.001
        batch_size = 400

        detector_noise = 1.00
        base_noise = 1.00
        nb_detector_samples = 100

    args = Args()
    eval_certify(detector, base, dataset, args)
