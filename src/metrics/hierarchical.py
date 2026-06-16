import numpy as np

from src.utils.core_utils import extend_subcat, intersection, get_top_level


def hierarchical_accuracy(subcat, predictions_gt, lambda_param=0.5):
    prediction_scores = []
    for prediction, gt in predictions_gt:
        if subcat == gt:
            try:
                prediction_top, prediction_sub = extend_subcat(prediction)
                gt_top, gt_sub = extend_subcat(gt)
            except Exception:
                # Unable to interpret hierarchical format for this label; skip
                continue

            if prediction_top == gt_top and prediction_sub == gt_sub:
                prediction_scores.append(1)
            elif prediction_top == gt_top and prediction_sub != gt_sub:
                prediction_scores.append(lambda_param)
            else:
                prediction_scores.append(0.0)

    if not prediction_scores:
        return 0.0

    class_acc = sum(prediction_scores) / len(prediction_scores)
    return class_acc


def hierarchical_prf_weighted(subcat, predictions_gt, lambda_param=0.75):
    hpp = []
    hrr = []
    for prediction, gt in predictions_gt:
        pi = extend_subcat(prediction)
        ti = extend_subcat(gt)
        pi_intersection_ti = intersection(pi, ti)

        if subcat == prediction:
            w = 1 if prediction == gt else (
                lambda_param if get_top_level(prediction) == get_top_level(gt) else 0
            )
            hp = (w * len(pi_intersection_ti)) / len(pi)
            hpp.append(hp)

        if subcat == gt:
            w = 1 if prediction == gt else (
                lambda_param if get_top_level(prediction) == get_top_level(gt) else 0
            )
            hr = (w * len(pi_intersection_ti)) / len(ti)
            hrr.append(hr)

    # If a class is never predicted (hpp empty), precision for that class should be 0
    # instead of raising ZeroDivision and being silently skipped upstream.
    class_p = (sum(hpp) / len(hpp)) if hpp else 0.0
    class_r = (sum(hrr) / len(hrr)) if hrr else 0.0
    if class_r == 0 and class_p == 0:
        class_f = 0
    else:
        class_f = 2 * class_p * class_r / (class_p + class_r)
    return class_p, class_r, class_f
