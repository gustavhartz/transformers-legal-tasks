from transformers.utils import logging
import json
import collections
import torch
from tqdm import tqdm
from utils import _get_best_indexes, _compute_softmax, get_final_text
from multiprocessing import Pool, cpu_count
from functools import partial
import time

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Very heavily inspired by the official evaluation script for SQuAD version 2.0 which was modified by XLNet authors to
update `find_best_threshold` scripts for SQuAD V2.0
In addition to basic functionality, we also compute additional statistics and plot precision-recall curves if an
additional na_prob.json file is provided. This file is expected to map question ID's to the model's predicted
probability that a question is unanswerable.
"""
"""
Modified version of "squad_metrics.py" adapated for CUAD.
"""


logger = logging.get_logger(__name__)


def compute_predictions_logits_multi_init(tokenizer_for_convert, json_input_dict_to_use, unique_id_to_result_to_use, example_index_to_features_to_use, contract_name_to_idx_to_use):
    global tokenizer
    tokenizer = tokenizer_for_convert
    global json_input_dict
    json_input_dict = json_input_dict_to_use
    global unique_id_to_result
    unique_id_to_result = unique_id_to_result_to_use
    global example_index_to_features
    example_index_to_features = example_index_to_features_to_use
    global contract_name_to_idx
    contract_name_to_idx = contract_name_to_idx_to_use


def compute_predictions_logits_multi(
    json_input_dict,
    all_examples,
    all_features,
    all_results,
    n_best_size,
    max_answer_length,
    do_lower_case,
    output_prediction_file,
    output_nbest_file,
    output_null_log_odds_file,
    verbose_logging,
    version_2_with_negative,
    null_score_diff_threshold,
    tokenizer,
    threads=2,
    chunk_size=2,
    tqdm_enabled=True,
):
    """Write final predictions to the json file and log-odds of null if needed."""
    if output_prediction_file:
        logger.info(f"Writing predictions to: {output_prediction_file}")
    if output_nbest_file:
        logger.info(f"Writing nbest to: {output_nbest_file}")
    if output_null_log_odds_file and version_2_with_negative:
        logger.info(f"Writing null_log_odds to: {output_null_log_odds_file}")

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    contract_name_to_idx = {}
    for idx in range(len(json_input_dict["data"])):
        contract_name_to_idx[json_input_dict["data"][idx]["title"]] = idx

    # Setup multi core
    threads = min(threads, cpu_count())
    time_start = time.time()
    with Pool(threads, initializer=compute_predictions_logits_multi_init, initargs=(tokenizer, json_input_dict, unique_id_to_result, example_index_to_features, contract_name_to_idx)) as p:
        annotate_ = partial(
            cuad_convert_example_output_to_prediction,
            n_best_size=n_best_size,
            max_answer_length=max_answer_length,
            do_lower_case=do_lower_case,
            verbose_logging=verbose_logging,
            version_2_with_negative=version_2_with_negative,
            null_score_diff_threshold=null_score_diff_threshold,
        )
        results = list(
            tqdm(
                p.imap(annotate_, enumerate(
                    all_examples), chunksize=chunk_size),
                total=len(all_examples),
                desc="Process examples to predictions",
                disable=not tqdm_enabled,
            )
        )
    time_end = time.time()
    logger.info(
        f"Processed {len(all_examples)} examples in {time_end - time_start:.1f} seconds")

    # Convert to dict
    all_predictions, all_nbest_json, scores_diff_json = collections.OrderedDict(
    ), collections.OrderedDict(), collections.OrderedDict()
    for pred, nbest, nulllog in results:
        all_predictions.update(pred)
        all_nbest_json.update(nbest)
        scores_diff_json.update(nulllog)

    if output_prediction_file:
        with open(output_prediction_file, "w") as writer:
            writer.write(json.dumps(all_predictions, indent=4) + "\n")

    if output_nbest_file:
        with open(output_nbest_file, "w") as writer:
            writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    if output_null_log_odds_file and version_2_with_negative:
        with open(output_null_log_odds_file, "w") as writer:
            writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

    return all_predictions


def cuad_convert_example_output_to_prediction(
    example,
    n_best_size,
    max_answer_length,
    do_lower_case,
    null_score_diff_threshold,
    version_2_with_negative,
    verbose_logging
):
    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction", ["feature_index", "start_index",
                             "end_index", "start_logit", "end_logit"]
    )

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()
    (example_index, example) = example

    features = example_index_to_features[example_index]

    contract_name = example.title
    contract_index = contract_name_to_idx[contract_name]
    paragraphs = json_input_dict["data"][contract_index]["paragraphs"]
    assert len(paragraphs) == 1

    prelim_predictions = []
    # keep track of the minimum score of null start+end of position 0
    score_null = 1000000  # large and positive
    min_null_feature_index = 0  # the paragraph slice with min null score
    null_start_logit = 0  # the start logit at the slice with min null score
    null_end_logit = 0  # the end logit at the slice with min null score
    for (feature_index, feature) in enumerate(features):
        result = unique_id_to_result[feature.unique_id]
        # TODO: This is a retarded way of doing this. It does not produce the same results top-n predictions. Two best indexes could be invalid
        start_indexes = _get_best_indexes(
            result.start_logits, n_best_size)
        end_indexes = _get_best_indexes(result.end_logits, n_best_size)
        # if we could have irrelevant answers, get the min score of irrelevant
        if version_2_with_negative:
            feature_null_score = result.start_logits[0] + \
                result.end_logits[0]
            if feature_null_score < score_null:
                score_null = feature_null_score
                min_null_feature_index = feature_index
                null_start_logit = result.start_logits[0]
                null_end_logit = result.end_logits[0]
        for start_index in start_indexes:
            for end_index in end_indexes:
                # We could hypothetically create invalid predictions, e.g., predict
                # that the start of the span is in the question. We throw out all
                # invalid predictions.
                if start_index >= len(feature.tokens):
                    continue
                if end_index >= len(feature.tokens):
                    continue
                if start_index not in feature.token_to_orig_map:
                    continue
                if end_index not in feature.token_to_orig_map:
                    continue
                if not feature.token_is_max_context.get(start_index, False):
                    continue
                if end_index < start_index:
                    continue
                length = end_index - start_index + 1
                if length > max_answer_length:
                    continue
                prelim_predictions.append(
                    _PrelimPrediction(
                        feature_index=feature_index,
                        start_index=start_index,
                        end_index=end_index,
                        start_logit=result.start_logits[start_index],
                        end_logit=result.end_logits[end_index],
                    )
                )
    if version_2_with_negative:
        prelim_predictions.append(
            _PrelimPrediction(
                feature_index=min_null_feature_index,
                start_index=0,
                end_index=0,
                start_logit=null_start_logit,
                end_logit=null_end_logit,
            )
        )
    prelim_predictions = sorted(prelim_predictions, key=lambda x: (
        x.start_logit + x.end_logit), reverse=True)

    _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "NbestPrediction", ["text", "start_logit", "end_logit"]
    )

    seen_predictions = {}
    nbest = []
    start_indexes = []
    end_indexes = []
    for pred in prelim_predictions:
        if len(nbest) >= n_best_size:
            break
        feature = features[pred.feature_index]
        if pred.start_index > 0:  # this is a non-null prediction
            tok_tokens = feature.tokens[pred.start_index: (
                pred.end_index + 1)]
            orig_doc_start = feature.token_to_orig_map[pred.start_index]
            orig_doc_end = feature.token_to_orig_map[pred.end_index]
            orig_tokens = example.doc_tokens[orig_doc_start: (
                orig_doc_end + 1)]

            tok_text = tokenizer.convert_tokens_to_string(tok_tokens)

            # Clean whitespace
            tok_text = tok_text.strip()
            tok_text = " ".join(tok_text.split())
            orig_text = " ".join(orig_tokens)

            final_text = get_final_text(
                tok_text, orig_text, do_lower_case, verbose_logging)

            if final_text in seen_predictions:
                continue

            seen_predictions[final_text] = True

            start_indexes.append(orig_doc_start)
            end_indexes.append(orig_doc_end)
        else:
            final_text = ""
            seen_predictions[final_text] = True

            start_indexes.append(-1)
            end_indexes.append(-1)

        nbest.append(_NbestPrediction(
            text=final_text, start_logit=pred.start_logit.item() if torch.is_tensor(pred.start_logit) else pred.start_logit, end_logit=pred.end_logit.item() if torch.is_tensor(pred.end_logit) else pred.end_logit))

    # if we didn't include the empty option in the n-best, include it
    if version_2_with_negative:
        if "" not in seen_predictions:
            nbest.append(_NbestPrediction(
                text="", start_logit=null_start_logit.item() if torch.is_tensor(null_start_logit) else null_start_logit, end_logit=null_end_logit.item() if torch.is_tensor(null_end_logit) else null_end_logit))
            start_indexes.append(-1)
            end_indexes.append(-1)

        # In very rare edge cases we could only have single null prediction.
        # So we just create a nonce prediction in this case to avoid failure.
        if len(nbest) == 1:
            nbest.insert(0, _NbestPrediction(
                text="empty", start_logit=0.0, end_logit=0.0))
            start_indexes.append(-1)
            end_indexes.append(-1)

    # In very rare edge cases we could have no valid predictions. So we
    # just create a nonce prediction in this case to avoid failure.
    if not nbest:
        nbest.append(_NbestPrediction(
            text="empty", start_logit=0.0, end_logit=0.0))
        start_indexes.append(-1)
        end_indexes.append(-1)

    assert len(nbest) >= 1, "No valid predictions"
    assert len(nbest) == len(start_indexes), "nbest length: {}, start_indexes length: {}".format(
        len(nbest), len(start_indexes))

    total_scores = []
    best_non_null_entry = None
    for entry in nbest:
        total_scores.append(entry.start_logit + entry.end_logit)
        if not best_non_null_entry:
            if entry.text:
                best_non_null_entry = entry

    probs = _compute_softmax(total_scores)

    nbest_json = []
    for (i, entry) in enumerate(nbest):
        output = collections.OrderedDict()
        output["text"] = entry.text
        output["probability"] = probs[i]
        output["start_logit"] = entry.start_logit.item() if torch.is_tensor(
            entry.start_logit) else entry.start_logit
        output["end_logit"] = entry.end_logit.item() if torch.is_tensor(
            entry.end_logit) else entry.end_logit
        output["token_doc_start"] = start_indexes[i]
        output["token_doc_end"] = end_indexes[i]
        nbest_json.append(output)

    assert len(nbest_json) >= 1, "No valid predictions"

    if not version_2_with_negative:
        all_predictions[example.qas_id] = nbest_json[0]["text"]
    else:
        # predict "" iff the null score - the score of best non-null > threshold
        score_diff = score_null - best_non_null_entry.start_logit - \
            (best_non_null_entry.end_logit)

        score_diff = score_diff.item() if torch.is_tensor(score_diff) else score_diff

        scores_diff_json[example.qas_id] = score_diff
        if score_diff > null_score_diff_threshold:
            all_predictions[example.qas_id] = ""
        else:
            all_predictions[example.qas_id] = best_non_null_entry.text
    all_nbest_json[example.qas_id] = nbest_json

    return all_predictions, all_nbest_json, scores_diff_json
