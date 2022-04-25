from ast import Tuple
import re
import string
from collections import Counter
from typing import List
import torch
import copy
import numpy as np
import os
from collections import namedtuple


def normalize_answer(s):
    '''
    Performs a series of cleaning steps on the ground truth and 
    predicted answer.
    '''
    s = copy.deepcopy(s)

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s)))).replace(" ", '')


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    '''
    Returns maximum value of metrics for predicition by model against
    multiple ground truths.
    :param func metric_fn: can be 'exact_match_score' or 'f1_score'
    :param str prediction: predicted answer span by the model
    :param list ground_truths: list of ground truths against which
                               metrics are calculated. Maximum values of 
                               metrics are chosen.
    '''
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)

    return max(scores_for_ground_truths)


def metric_max_over_ground_truth(metric_fn, prediction, ground_truth):
    '''
    Returns maximum value of metrics for predicition by model against
    a single ground truth.
    :param func metric_fn: can be 'exact_match_score' or 'f1_score'
    :param str prediction: predicted answer span by the model
    :param str ground_truth: ground truth answer span
    '''
    return metric_fn(prediction, ground_truth)

# Function that takes list of pairs of answers and ground truth an calculates metrics


def evaluate(answers, ground_truths):
    '''
    Calculates exact_match_score, f1_score, and precision_score for a list of
    predicted answers and ground truths.
    '''

    exact_match = []
    f1_scores = []
    for answer, ground_truth in zip(answers, ground_truths):
        exact_match.append(exact_match_score(answer, ground_truth))
        f1_scores.append(f1_score(answer, ground_truth))
    return {
        'exact_match': 100.0 * float(sum(exact_match)) / len(exact_match),
        'f1': 100.0 * float(sum(f1_scores)) / len(f1_scores)
    }


def f1_score(prediction, ground_truth):
    '''
    Returns f1 score of two strings.
    '''
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    '''
    Returns exact_match_score of two strings.
    '''
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

# Given the batch, output and tokenizer return f1 and exact match scores


def calculate_batch_accuracy(batch, start, end, tokenizer):
    '''
    Calculates exact_match_score and f1_score for a batch of predictions and
    ground truths.
    '''

    batch_values = list(zip(
        batch['input_ids'], batch['token_type_ids'], batch['start_positions'], batch['end_positions']))
    # Non zero values correspond to the seperation
    # https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertTokenizer
    answers = [tokenizer.decode(x[0][x[2]:x[3]]) for x in batch_values]

    # Only top 1 prediction
    start_pred = torch.argmax(start, dim=1)
    end_pred = torch.argmax(end, dim=1)
    # decode
    preds_text = [tokenizer.decode(x[2][x[0]:x[1]]) if x[0] < x[1] else "[END BEFORE START]" for x in zip(
        start_pred, end_pred, batch['input_ids'])]
    return evaluate(answers, preds_text)


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


def compute_top_1_scores_from_preds(predictions, iou_threshold=0.5, include_df=False):
    """Generates the top 1 predictions from the given predictions in the get_predictions_from_batch function.

    Args:
        predictions (_type_): output of the get_predictions_from_batch function
        iou_threshold (float, optional): jaccard limit. Defaults to 0.5.
        include_df (bool, optional): get df of results. Defaults to False.

    Returns:
        _type_: _description_
    """

    # Compute precision, recall, f1, em - best
    df = []
    df_struct = {"id": None,
                 "question": None,
                 "is_impossible": None,
                 "pred": None,
                 "answer": None,
                 "jaccard": None,
                 "doc_length": None,
                 "em": None,
                 "tp": None,
                 "fp": None,
                 "tn": None,
                 "fn": None,
                 "top_k_idx": None,
                 "confidence": None,
                 "start_token_loc_pred": None,
                 "end_token_loc_pred": None}
    IOU_THRESH = iou_threshold

    # Text ans pred text, imp ans pred text, text ans pred imp, imp ans pred imp
    tp, fp, fn, tn = 0, 0, 0, 0
    # exact match
    em = 0
    for prediction_set in predictions:

        # Currently only top 1
        _id, top_k_idx, is_impossible, _pred_text, answer, confidence, start_loc_pred, end_loc_pred, feature_index = prediction_set[
            0]
        _tmp_df = copy.copy(df_struct)

        # Set values
        _tmp_df["is_impossible"] = is_impossible
        _tmp_df["top_k_idx"] = 0
        _tmp_df["id"] = _id
        _tmp_df["answer"] = answer
        _tmp_df["pred"] = _pred_text
        _tmp_df["confidence"] = confidence
        _tmp_df["start_token_loc_pred"] = start_loc_pred
        _tmp_df["end_token_loc_pred"] = end_loc_pred

        if is_impossible:
            if len(_pred_text) > 0:
                _tmp_df['fp'] = True
                fp += 1
            else:
                tn += 1
                _tmp_df['tn'] = True
        else:
            if len(_pred_text) < 1:
                fn += 1
                _tmp_df['fn'] = True
            else:
                jc = get_jaccard(answer, _pred_text)
                _tmp_df['jaccard'] = jc
                if normalize_answer(answer) == normalize_answer(_pred_text):
                    _tmp_df['em'] = True
                    _tmp_df['tp'] = True
                    em += 1
                    tp += 1
                elif jc >= IOU_THRESH:
                    _tmp_df['tp'] = True
                    tp += 1
                else:
                    _tmp_df['fp'] = True
                    fp += 1
        df.append(_tmp_df)
    precision = (tp+tn) / (tp+tn + fp) if tp+tn + fp > 0 else np.nan
    recall = (tp+tn) / (tp+tn + fn) if tp+tn + fn > 0 else np.nan

    res = {"tp": tp,
           "fp": fp,
           "fn": fn,
           "tn": tn,
           "em": em,
           "acc": (tp+tn) / (tp+tn + fp + fn),
           "precision": precision,
           "recall": recall,
           "batch_len": len(predictions),
           "f1": 2*(precision*recall)/(precision+recall) if precision+recall > 0 else np.nan}
    if include_df:
        return res, df
    return res


def feature_path(args, feature_index):
    """Generates the feature path and prefix for the given args. Combine with the feature_path_and_prefix function."""
    DATASET_NAME = args.dataset_name+"_"+args.model_type
    dataset_path = os.path.join(
        args.out_dir, "features", DATASET_NAME + "_eval_" + args.predict_file_version + f"_{feature_index}_features_validation")
    return dataset_path


def get_pred_from_batch_outputs(args, batch, start_logits, end_logits, tokenizer, top_k=2, max_ans_len=200) -> List[List[Tuple]]:
    """Takes as input the batch and the outputs of the model and returns the predictions

    Args:
        args (): arguments
        batch (_type_): Batch from dataloader
        start_logits (_type_): start logits from the model
        end_logits (_type_): end logits from the model
        tokenizer (_type_): To decode the tokens
        top_k (int, optional): how many predictions to return. Defaults to 2.
        max_ans_len (int, optional): limit answers to specific lenght. Defaults to 200.

    Returns:
        List[List[Tuple]]: Batch x Top k predictions x (id, idx_top_k , is_impossible ,_pred_text,_answer_text,confidence, _start, _end)
    """
    _PrelimPrediction = namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction", ["feature_unique_id", "top_k_idx", "is_impossible",
                             "pred_text", "answer_text", "confidence", "start_index", "end_index", "feature_index"]
    )
    # Get logits
    start_ = start_logits.detach()
    end_ = end_logits.detach()
    feat_ind = batch[-1].detach()
    # Normalise
    start_ = torch.exp(
        start_ - torch.log(torch.sum(torch.exp(start_), axis=-1, keepdims=True)))
    end_ = torch.exp(
        end_ - torch.log(torch.sum(torch.exp(end_), axis=-1, keepdims=True)))

    # List of batch predictions
    p = [decode(x, y, top_k, max_ans_len) for x, y in zip(start_, end_)]

    predictions = []

    i = 0
    # Batch results
    for ele in p:
        # Top k
        temp_collect = []
        # Each single predictions
        feat = torch.load(feature_path(args, feat_ind[i]))
        for idx, _v in enumerate(zip(ele[0], ele[1])):
            _start, _end = _v
            _pred = batch[0][i][_start:_end+1]
            _id = feat.unique_id
            _answer_start = feat.start_position
            _answer_end = feat.end_position

            _answer = batch[0][i][_answer_start:_answer_end+1]

            confidence = ele[2][0][idx]
            is_impossible = feat.is_impossible

            # get answer and pred text
            if (_start == _end == 1) or (_end == 0) or feat.is_impossible:
                is_impossible = True
                _pred_text = ""
            else:
                _pred_text = tokenizer.decode(_pred)

            if (_answer_start == _answer_end == 1) or (is_impossible) or (_answer_start == 0):
                _answer_text = ""
            else:
                _answer_text = tokenizer.decode(_answer)

            temp_collect.append(
                _PrelimPrediction(
                    feature_unique_id=_id,
                    top_k_idx=idx,
                    is_impossible=is_impossible,
                    pred_text=_pred_text,
                    answer_text=_answer_text,
                    confidence=confidence,
                    start_index=_start,
                    end_index=_end,
                    feature_index=i
                )
            )
        predictions.append(temp_collect)
        i += 1
    return predictions


def decode(
    start: torch.tensor, end: torch.tensor, topk: int, max_answer_len: int
):
    """
    Take the output of any `ModelForQuestionAnswering` and will generate probabilities for each span to be the
    actual answer.
    In addition, it filters out some unwanted/impossible cases like answer len being greater than max_answer_len or
    answer end position being before the starting position. The method supports output the k-best answer through
    the topk argument.
    Args:
        start (`torch.tensor`): Individual start probabilities for each token.
        end (`torch.tensor`): Individual end probabilities for each token.
        topk (`int`): Indicates how many possible answer span(s) to extract from the model output.
        max_answer_len (`int`): Maximum size of the answer to extract from the model's output.
    """
    # Ensure we have batch axis
    if start.ndim == 1:
        start = start[None]

    if end.ndim == 1:
        end = end[None]

    # Compute the score of each tuple(start, end) to be the real answer
    outer = torch.matmul(start.unsqueeze(-1), end.unsqueeze(1))

    # Remove candidate with end < start and end - start > max_answer_len
    candidates = torch.tril(torch.triu(outer), max_answer_len - 1)
    #  Inspired by Chen & al. (https://github.com/facebookresearch/DrQA)
    scores_flat = candidates.flatten()
    # Get nr. 1
    if topk == 1:
        idx_sort = [torch.argmax(scores_flat)]
    elif len(scores_flat) < topk:
        idx_sort = torch.argsort(-scores_flat)
    else:
        idx = torch.topk(scores_flat, topk).indices
        idx_sort = idx[torch.argsort(-scores_flat[idx])]

    def unravel_index(index, shape):
        out = []
        for dim in reversed(shape):
            out.append(index % dim)
            index = torch.div(index, dim, rounding_mode='trunc')
        return tuple(reversed(out))
    starts, ends = unravel_index(idx_sort, candidates.shape)[1:]
    scores = candidates[:, starts, ends]

    return starts, ends, scores


# Test functions if name == __main__
if __name__ == "__main__":
    # List of ground truths to 10 random questions
    gt_list = ['What is the capital of France?',
               'What is the capital of Germany?',
               'What is the capital of Italy?',
               'What is the capital of Spain?',
               'What is the capital of the United States?',
               'What is the capital of the United Kingdom?',
               'What is the capital of the United States?',
               'What is the capital of the United Kingdom?']
    # Answers to the questions
    answers = ['What is the capital of France?', 'the capital of Germany?',
               'the capital of Italy?', '  easf the capital of Spain?',
               'the capital of the United States? wda',
               'the capital of the United Kingdom?',
               'the capital of the United States?',
               'the capital of the United Kingdom?']
    # Calculate the metrics
    print(evaluate(answers, gt_list))
