import time
import unittest

import numpy as np
from sklearn.metrics import matthews_corrcoef, confusion_matrix

from lib.rewards import correct_predictions
from lib.visualizer import format_matrix, pprint, color_mask


class TestPredictions(unittest.TestCase):
    def test_correction(self):
        np.random.seed(0)

        y_labels = np.random.binomial(1, p=0.02, size=10000)
        y_preds = y_labels.copy()

        # Make predictions
        mask = np.zeros_like(y_preds, dtype=np.bool)
        rand_ints = np.random.randint(0, len(y_preds), len(y_preds) // 5)
        mask[rand_ints] = True
        y_preds[mask] = 1 - y_preds[mask]

        # Correct predictions
        out_preds = correct_predictions(y_labels, y_preds, w_f=0, w_b=5)

        pprint("mask", color_mask(mask, mask) + "\n", shift=10)
        pprint("y", format_matrix(y_labels)[0], shift=10)
        pprint("y_preds", format_matrix(y_preds)[0], shift=10)

        pprint(
            "correction",
            color_mask(
                np.not_equal(y_preds, out_preds),
                np.not_equal(y_preds, out_preds),
                color=1,
            ),
            shift=10,
        )
        pprint("out_preds", format_matrix(out_preds)[0], shift=10)
        pprint("y", format_matrix(y_labels)[0], shift=10)

        pprint("\t- mcc", "{:.3f}".format(matthews_corrcoef(y_labels, y_preds)))
        pprint(
            "\t- mcc_out",
            "{:.3f}".format(matthews_corrcoef(y_labels, out_preds)) + "\n",
        )

        c_matrix = confusion_matrix(y_labels, out_preds)
        tn = c_matrix[0][0]
        fn = c_matrix[1][0]
        tp = c_matrix[1][1]
        fp = c_matrix[0][1]

        pprint("\t- TP:", "{:.3f}".format(tp / (tp + fn)))
        pprint("\t- FN:", "{:.3f}".format(fn / (tp + fn)))
        pprint("\t- FP:", "{:.3f}".format(fp / (tn + fp)))
        pprint("\t- TN:", "{:.3f}".format(tn / (tn + fp)))

        time.sleep(0.1)
        self.assertTrue(True)
