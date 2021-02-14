import numpy as np
import pandas as pd


class RewardL2RPN2019:
    def from_observation(self, obs):
        relative_flows = obs.rho
        print("obs", relative_flows)
        reward = self.from_relative_flows(relative_flows)
        return reward

    def from_mip_solution(self, result):
        line_flow = pd.concat(
            [result["res_line"]["p_pu"], result["res_trafo"]["p_pu"]], ignore_index=True
        )
        max_line_flow = pd.concat(
            [result["res_line"]["max_p_pu"], result["res_trafo"]["max_p_pu"]],
            ignore_index=True,
        )

        relative_flows = np.abs(
            np.divide(line_flow, max_line_flow + 1e-9)
        )  # rho_l = abs(F_l / F_l^max)
        relative_flows = relative_flows * np.greater(relative_flows, 1e-9).astype(float)

        reward = self.from_relative_flows(relative_flows)
        return reward

    @staticmethod
    def from_relative_flows(relative_flows):
        relative_flows = np.minimum(relative_flows, 1.0)  # Clip if rho > 1.0

        line_scores = np.maximum(
            1.0 - relative_flows ** 2, np.zeros_like(relative_flows)
        )

        reward = line_scores.sum()
        return reward


def correct_predictions(y_labels, y_preds, w_f=0, w_b=0):
    out_preds = y_preds.copy()

    for t, (y_label, y_pred) in enumerate(zip(y_labels, y_preds)):
        if y_label != y_pred:

            # Count number of positive labels in the next w_f time steps
            window_pos_forward_count = np.sum(y_labels[(t + 1) : (t + w_f + 1)])
            if y_pred == 1 and window_pos_forward_count > 0:
                out_preds[t] = y_label
                continue

            window_preds_backward_count = np.sum(y_preds[(t - w_b) : t])
            if y_pred == 0 and window_preds_backward_count > 0:
                out_preds[t] = y_label
                continue

    return out_preds
