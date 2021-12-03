# coding=utf-8
import torch
from scipy.stats import norm, binom_test
import numpy as np
from math import ceil
from statsmodels.stats.proportion import proportion_confint

# to abstain, Smooth returns this int
ABSTAIN = -1


class CalibrateDetector(object):
    def __init__(self, base_classifier, original_num_classes, num_classes, sigma):
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.original_num_classes = original_num_classes
        self.sigma = sigma

    def predict(self, x, n, alpha, batch_size, penalize_weight):
        self.base_classifier.eval()
        counts, calibrated_label = self.sample_noise(x, n, batch_size, penalize_weight)
        top2 = counts.argsort()[::-1][:2]
        count1 = counts[top2[0]]
        count2 = counts[top2[1]]
        if binom_test(count1, count1 + count2, p=0.5) > alpha:
            return ABSTAIN, calibrated_label
        else:
            return top2[0], calibrated_label

    def sample_noise(self, x,  num, batch_size, penalize_weight):
        outputs = x
        counts = np.zeros(self.num_classes, dtype=int)
        calibrated_prob = np.zeros(10, dtype=int)
        for _ in range(ceil(num / batch_size)):
            this_batch_size = min(batch_size, num)
            num -= this_batch_size
            batch = outputs.repeat((this_batch_size, 1))
            noise = torch.randn_like(batch, device='cuda') * self.sigma
            noisy_batch = batch + noise
            predictions = self.base_classifier(noisy_batch).argmax(1).cpu().numpy()
            counts += self.count_arr(predictions, self.num_classes)
            noisy_batch = noisy_batch.cpu().numpy()
            labels = np.argmax(noisy_batch, axis=1)
            np.add.at(calibrated_prob, labels, 1)
        calibrated_prob = calibrated_prob * penalize_weight
        calibrated_label = np.argmax(calibrated_prob)
        return counts, calibrated_label

    @staticmethod
    def count_arr(arr: np.ndarray, length: int) -> np.ndarray:
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            counts[idx] += 1
        return counts


class CalibrateConfidenceSmooth(object):
    def __init__(self, base_classifier: torch.nn.Module, detector: CalibrateDetector, num_classes: int, sigma: float,
                 nb_detector: int):
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.sigma = sigma
        self.detector = detector
        self.detector_sigma = detector.sigma
        self.flag = False
        self.penalize_weights = []
        self.nb_detector = nb_detector

    def certify(self, x: torch.tensor, n0: int, n: int, alpha: float, batch_size: int) -> (int, float):
        self.base_classifier.eval()
        counts_selection = self.sample_noise_pure(x, n0, batch_size)
        self.penalize_weights = counts_selection / n0
        cAHat = counts_selection.argmax().item()
        self.penalize_weights[cAHat] += 2
        counts_estimation = self.sample_noise(x, n, batch_size)
        nA = counts_estimation[cAHat].item()
        pABar = self._lower_confidence_bound(nA, n, alpha)
        if pABar < 0.5:
            return ABSTAIN, 0.0
        else:
            radius = self.sigma * norm.ppf(pABar)
            return cAHat, radius

    def sample_noise_pure(self, x: torch.tensor, num: int, batch_size) -> np.ndarray:
        with torch.no_grad():
            counts = np.zeros(self.num_classes, dtype=int)
            for _ in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size

                batch = x.repeat((this_batch_size, 1, 1, 1))
                noise = torch.randn_like(batch, device='cuda') * self.sigma
                predictions = self.base_classifier(batch + noise).argmax(1)
                counts += self.count_arr(predictions.cpu().numpy(), self.num_classes)
            return counts

    def sample_noise(self, x, num, batch_size):
        with torch.no_grad():
            counts = np.zeros(self.num_classes, dtype=int)
            # counts_orig = np.zeros(self.num_classes, dtype=int)
            for _ in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size

                batch = x.repeat((this_batch_size, 1, 1, 1))
                noise = torch.randn_like(batch, device='cuda') * self.sigma
                outputs = self.base_classifier(batch + noise)

                detector_rs = [self.detector.predict(output, n=self.nb_detector, alpha=0.001, batch_size=400,
                                                     penalize_weight=self.penalize_weights) for output in
                               outputs]
                counts += self.count(outputs.cpu().numpy().tolist(), detector_rs, self.num_classes)
        return counts

    @staticmethod
    def count(arr, detector_rs, length):
        counts = np.zeros(length, dtype=int)
        for i in range(len(arr)):
            base_output = np.asarray(arr[i])
            is_stable, calibrated_label = detector_rs[i]
            if is_stable == 0:
                pred = np.argmax(base_output)
            else:
                pred = calibrated_label
            counts[pred] += 1
        return counts

    @staticmethod
    def count_arr(arr: np.ndarray, length: int) -> np.ndarray:
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            counts[idx] += 1
        return counts

    @staticmethod
    def _lower_confidence_bound(NA: int, N: int, alpha: float) -> float:
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]