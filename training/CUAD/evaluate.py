import json
import pandas as pd
import numpy as np
from sklearn import metrics

IOU_THRESH = 0.5


def get_questions_from_csv(file_path):
    df = pd.read_csv(file_path)
    q_dict = {}
    for i in range(df.shape[0]):
        category = df.iloc[i, 0].split("Category: ")[1]
        description = df.iloc[i, 1].split("Description: ")[1]
        q_dict[category.title()] = description
    return q_dict


def load_json(path):
    with open(path, "r") as f:
        dict = json.load(f)
    return dict


def get_preds(nbest_preds_dict, conf=None):
    results = {}
    for question_id in nbest_preds_dict:
        list_of_pred_dicts = nbest_preds_dict[question_id]
        preds = {}
        for pred_dict in list_of_pred_dicts:
            text = pred_dict["text"]
            prob = pred_dict["probability"]
            if not text == "":  # don't count empty string as a prediction
                preds[text] = prob
        preds_list = [pred for pred in preds.keys() if preds[pred] > conf]
        results[question_id] = preds_list
    return results


def get_answers(test_json_dict):
    results = {}

    data = test_json_dict["data"]
    for contract in data:
        for para in contract["paragraphs"]:
            qas = para["qas"]
            for qa in qas:
                id = qa["id"]
                answers = qa["answers"]
                answers = [answers[i]["text"] for i in range(len(answers))]
                results[id] = answers

    return results


def get_jaccard(gt, pred):
    remove_tokens = [".", ",", ";", ":"]
    for token in remove_tokens:
        gt = gt.replace(token, "")
        pred = pred.replace(token, "")
    gt = gt.lower()
    pred = pred.lower()
    gt = gt.replace("/", " ")
    pred = pred.replace("/", " ")

    gt_words = set(gt.split(" "))
    pred_words = set(pred.split(" "))

    intersection = gt_words.intersection(pred_words)
    union = gt_words.union(pred_words)
    jaccard = len(intersection) / len(union)
    return jaccard


def compute_precision_recall(gt_dict, preds_dict, category=None):
    tp, fp, fn = 0, 0, 0

    for key in gt_dict:
        if category and category not in key:
            continue

        substr_ok = "Parties" in key

        answers = gt_dict[key]
        preds = preds_dict[key]

        # first check if answers is empty
        if len(answers) == 0:
            if len(preds) > 0:
                fp += len(preds)  # false positive for each one
        else:
            for ans in answers:
                assert len(ans) > 0
                # check if there is a match
                match_found = False
                for pred in preds:
                    if substr_ok:
                        is_match = get_jaccard(
                            ans, pred) >= IOU_THRESH or ans in pred
                    else:
                        is_match = get_jaccard(ans, pred) >= IOU_THRESH
                    if is_match:
                        match_found = True

                if match_found:
                    tp += 1
                else:
                    fn += 1

            # now also get any fps by looping through preds
            for pred in preds:
                # Check if there's a match. if so, don't count (don't want to double count based on the above)
                # but if there's no match, then this is a false positive.
                # (Note: we get the true positives in the above loop instead of this loop so that we don't double count
                # multiple predictions that are matched with the same answer.)
                match_found = False
                for ans in answers:
                    assert len(ans) > 0
                    if substr_ok:
                        is_match = get_jaccard(
                            ans, pred) >= IOU_THRESH or ans in pred
                    else:
                        is_match = get_jaccard(ans, pred) >= IOU_THRESH
                    if is_match:
                        match_found = True

                if not match_found:
                    fp += 1

    precision = tp / (tp + fp) if tp + fp > 0 else np.nan
    recall = tp / (tp + fn) if tp + fn > 0 else np.nan

    return precision, recall


def process_precisions(precision):
    """
    Processes precisions to ensure that precision and recall don't both get worse
    Assumes the list precision is sorted in order of recalls
    """
    precision_best = precision[::-1]
    for i in range(1, len(precision_best)):
        precision_best[i] = max(precision_best[i-1], precision_best[i])
    precision = precision_best[::-1]
    return precision


def get_prec_at_recall(precisions, recalls, confs, recall_thresh=0.9):
    """
    Assumes recalls are sorted in increasing order
    """
    processed_precisions = process_precisions(precisions)
    prec_at_recall = 0
    for prec, recall, conf in zip(processed_precisions, recalls, confs):
        if recall >= recall_thresh:
            prec_at_recall = prec
            break
    return prec_at_recall, conf


def get_precisions_recalls(pred_dict, gt_dict, category=None):
    precisions = [1]
    recalls = [0]
    confs = []
    for conf in list(np.arange(0.99, 0, -0.01)) + [0.001, 0]:
        conf_thresh_pred_dict = get_preds(pred_dict, conf)
        prec, recall = compute_precision_recall(
            gt_dict, conf_thresh_pred_dict, category=category)
        precisions.append(prec)
        recalls.append(recall)
        confs.append(conf)
    return precisions, recalls, confs


def get_aupr(precisions, recalls):
    processed_precisions = process_precisions(precisions)
    aupr = metrics.auc(recalls, processed_precisions)
    if np.isnan(aupr):
        return 0
    return aupr


def get_results(args, n_best_predictions_path, gt_dict, gt_dict_extract_answers=True, include_model_info=True, filter=None):
    """

    Args:
        args (_type_): _description_
        n_best_predictions_path (_type_): _description_
        gt_dict (_type_): _description_
        gt_dict_extract_answers (bool, optional): _description_. Defaults to True.
        include_model_info (bool, optional): _description_. Defaults to True.
        filter (_type_, optional): Filters the results to keep a specific subset of questions. Defaults to None.

    Returns:
        _type_: _description_
    """

    gt_dict = get_answers(gt_dict) if gt_dict_extract_answers else gt_dict

    pred_dict = load_json(n_best_predictions_path)

    assert sorted(list(pred_dict.keys())) == sorted(list(gt_dict.keys()))

    # Remove examples with empty answers
    to_rm = []
    if filter == "No Ans":
        to_rm = [k for k, v in gt_dict.items() if not v]
        print("Removing {} examples with empty answers".format(len(to_rm)))
    elif filter == "Has Ans":
        to_rm = [k for k, v in gt_dict.items() if v]
        print("Removing {} examples with answers".format(len(to_rm)))
    elif filter:
        to_rm = [k for k, v in gt_dict.items() if filter not in k]
        print("Removing {} examples with answers".format(len(to_rm)))
    for k in to_rm:
        gt_dict.pop(k, None)
        pred_dict.pop(k, None)

    assert sorted(list(pred_dict.keys())) == sorted(list(gt_dict.keys()))

    precisions, recalls, confs = get_precisions_recalls(pred_dict, gt_dict)
    prec_at_90_recall, _ = get_prec_at_recall(
        precisions, recalls, confs, recall_thresh=0.9)
    prec_at_80_recall, _ = get_prec_at_recall(
        precisions, recalls, confs, recall_thresh=0.8)
    aupr = get_aupr(precisions, recalls)

    # now save results as a dataframe and return
    results = {"name": args.model_name, "version": args.model_version, "aupr": aupr, "prec_at_80_recall": prec_at_80_recall,
               "prec_at_90_recall": prec_at_90_recall}
    if not include_model_info:
        del results["name"]
        del results["version"]
    return results
