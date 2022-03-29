from datasets import load_dataset
import random
import re
import string
import copy
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer
import argparse
import json
import pandas as pd
import torch

# Function that generates random slice a list with len CONTEXT_LEN_TOKEN that includes the tkn_start and tkn_end positions
# Does have some issues if the answer is at the end of the context


def random_slice_index(l, tkn_start, tkn_end, slice_len=500):
    if len(l) <= slice_len:
        return (0, len(l)-2)

    # Random start position some place before the start token but with space to cover the answer span

    # At begining of inputs or slice_len before end so we know there is an overlap
    start_loc = max(tkn_end-slice_len, 0)
    start = random.randint(start_loc, tkn_start)

    # Need to be within array
    # Zero indexed and the end element causes issues with token_to_ids
    end = min(start + slice_len, len(l)-2)
    return (start, end)


def check_match(str1, str2):
    if len(str1) > len(str2):
        str1, str2 = str2, str1
    count = 0
    i = 0
    while (i < len(str1)) and (i+1 < len(str2)):
        if str1[i] == str2[i]:
            count += 1
        elif str1[i] == str2[i+1]:
            count += 1
        elif str1[i] == str2[i-1]:
            count += 1
        i += 1
    return count


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

    return white_space_fix(remove_punc(lower(s))).replace(" ", '').replace('\xad', '').replace('\u00ad', '').replace('\N{SOFT HYPHEN}', '').replace('\u200b', '')


def create_dataset(ds, tokenizer, _len, token_len=500):
    train = ds
    _errs = {"too long": 0,
             "no ans": 0,
             "weird string": 0,
             "Multiple Answers": 0}
    res = []
    id_map = {}
    _i = 0
    for sample in tqdm(train):
        id_map[_i] = sample['id']
        _i += 1
        # Get data entries
        title = sample['title']
        context = sample['context']
        question = sample['question']
        answers = sample['answers']

        # Tokenize question to get lenght
        q_tok = tokenizer(question)

        if not answers['text']:
            _errs['no ans'] += 1
            # get random section
            start_pos = random.randint(
                0, max(len(context)-_len*4, min(_len*4, max(len(context)-token_len, 0))))
            end_pos = start_pos + _len*4
            context_slice = context[start_pos:end_pos]

            # Ensure a large sample is inserted
            encoding = tokenizer(context_slice, question,
                                 truncation=True, padding=True)
            start = encoding['token_type_ids'].index(1)-2
            # yield random answer
            res.append({'start_positions': 1,
                        'end_positions': 1,
                        'question': question,
                        'context': context_slice[:encoding.token_to_chars(start).end],
                        'id': _i,
                        'original_id': sample['id'],
                        'char_span_start': start_pos,
                        'is_impossible': True,
                        'title': title,
                        'answer': ''})
            continue
        if len(answers['text']) > 1:
            _errs['Multiple Answers'] += 1
            continue
        start_pos = answers['answer_start'][0]
        end_pos = start_pos + len(answers['text'][0])
        start_offset = max(0, start_pos-_len*2)
        end_offset = min(len(context), end_pos+_len*2)
        start_pos_small_context = start_pos - start_offset
        end_pos_small_context = end_pos - start_offset

        # Manipulate data
        small_context = context[start_offset:end_offset]
        sml_ctx_start = start_offset
        # Get tokenization data
        encoding = tokenizer(small_context)
        tkn_start = encoding.char_to_token(start_pos_small_context)
        # Minus one to hit the last char of the workd
        tkn_end = encoding.char_to_token(end_pos_small_context-1)

        # Ensure that the answer can be fit into the context
        if tkn_start+450 < tkn_end:
            _errs['too long'] += 1
            continue

        # Test we have the right answer
        detokenised_answer = tokenizer.decode(
            encoding['input_ids'][tkn_start:tkn_end+1])  # + one to get inclusive last word
        # Needs a bit more processing
        s1 = normalize_answer(answers['text'][0])
        s2 = normalize_answer(detokenised_answer)
        try:
            assert (s1 == s2) or (check_match(s1, s2) > 0.8*len(s1))
        except:
            _errs['weird string'] += 1
            continue

        """
        We can now reduce the context to max size of CONTEXT_LEN_TOKEN and using a randomised approach

        We know that all tokens between tkn_start and tkn_end inclusive needs to be contained in the encoding.

        The approarch we use is:
        * select a random overlap window over the input_ids including our answer of the desired size
        * Convert this back to positions in the original small_context
        * Recompute the positions


        """
        # Get random split
        if tkn_end-tkn_start + len(q_tok['input_ids']) > token_len-10:
            _errs['no ans'] += 1
            continue
        # TODO: This might be the lenght issue
        loc = random_slice_index(encoding['input_ids'], tkn_start,
                                 tkn_end, token_len-10 - len(q_tok['input_ids']))
        new_char_span = (encoding.token_to_chars(max(loc[0], 1))[
            0], encoding.token_to_chars(loc[1])[1])

        # We can now create the final data
        pre_pro_context = small_context[new_char_span[0]:new_char_span[1]+1]
        pre_pro_ans_start_pos = start_pos_small_context-new_char_span[0]
        pre_pro_ans_end_pos = end_pos_small_context-new_char_span[0]
        # Check answers allign
        s2 = normalize_answer(
            pre_pro_context[pre_pro_ans_start_pos:pre_pro_ans_end_pos])
        assert (s1 == s2) or (check_match(s1, s2) > 0.8*len(s1))

        res.append({'start_positions': pre_pro_ans_start_pos,
                    'end_positions': pre_pro_ans_end_pos,
                    'question': question,
                    'context': pre_pro_context,
                    'id': _i,
                    'original_id': sample['id'],
                    'char_span_start': sml_ctx_start+new_char_span[0],
                    'is_impossible': False,
                    'answer': answers['text'][0],
                    'title': title})
    print("Errors:", _errs)
    return res, id_map


def add_token_positions(encodings, answers, tokenizer):
    start_positions = []
    end_positions = []
    for i in range(len(answers)):
        # Prevent issue with empty answers getting wrong span
        if answers[i][0] == answers[i][1] == 1:
            start_positions.append(1)
            end_positions.append(1)
        else:
            start_positions.append(encodings.char_to_token(
                i, answers[i][0]))
            end_positions.append(encodings.char_to_token(
                i, answers[i][1] - 1))
            # if None, the answer passage has been truncated meaning going outside this current span
            if start_positions[-1] is None:
                start_positions[-1] = tokenizer.model_max_length
            if end_positions[-1] is None:
                end_positions[-1] = tokenizer.model_max_length
    encodings.update({'start_positions': start_positions,
                     'end_positions': end_positions})


def create_encoding_dataset(df, tokenizer):
    train_encodings = tokenizer(
        list(df.context), list(df.question), truncation=True, padding=True)
    train_encodings.update({'id': list(df.id), 'char_span_start': list(
        df.char_span_start), 'is_impossible': list(df.is_impossible)})
    add_token_positions(train_encodings, list(
        zip(list(df.start_positions), list(df.end_positions))), tokenizer)
    return train_encodings


def main(args):
    train = load_dataset("cuad", split='train')
    test = load_dataset("cuad", split='test')
    tokenizer = AutoTokenizer.from_pretrained(args.model_type)
    config = AutoConfig.from_pretrained(
        args.model_type,
        cache_dir=None,
    )
    _len = args.contex_size_snippet
    token_limit = config.max_position_embeddings
    random.seed(5132)
    # Not very pretty
    print("Creating train set")
    train, train_id_map = create_dataset(train, tokenizer, _len, token_limit)
    print("Creating train set")
    test, test_id_map = create_dataset(test, tokenizer, _len, token_limit)
    print("Saving data to cwd set")
    # Add postfix to avoid overwriting

    with open(f'./data/train_data{args.postfix}.json', 'w') as fout:
        json.dump({'data': train}, fout)
    with open(f'./data/train_data_id_mapping{args.postfix}.json', 'w') as fout:
        json.dump({'data': train_id_map}, fout)
    with open(f'./data/test_data.json{args.postfix}', 'w') as fout:
        json.dump({'data': test}, fout)
    with open(f'./data/test_data_id_mapping{args.postfix}.json', 'w') as fout:
        json.dump({'data': test_id_map}, fout)

    # Tokenize data if arg
    if args.create_encodings:
        print("Tokenizing data")
        train_encodings = create_encoding_dataset(
            pd.DataFrame(train), tokenizer)
        test_encodings = create_encoding_dataset(pd.DataFrame(test), tokenizer)

        torch.save(train_encodings, "./data/train_encodings")
        torch.save(test_encodings, "./data/test_encodings")
        # Pickle dump encodings
    print('Done!')


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='Process hyper-parameters')
    # Argument for model type
    argparser.add_argument('--model_type', type=str,
                           default='Rakib/roberta-base-on-cuad', help='Model type to use')
    # Model max sequence length
    argparser.add_argument('--max_seq_length', type=int,
                           default=512, help='Max sequence length')
    # Model max sequence length soft limit
    argparser.add_argument('--contex_size_snippet', type=int,
                           default=1500, help='Max sequence length to aim for')

    args = argparser.parse_args()
    main(args)
