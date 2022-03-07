import re
import string
from collections import Counter
import torch
import copy


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
