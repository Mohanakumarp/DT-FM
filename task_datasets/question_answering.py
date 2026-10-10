"""SQuAD windows, answer-span labels, and example-level exact match/F1."""
from collections import Counter
import re
import string


def tokenize_qa(examples, tokenizer, max_length, stride):
    encoded = tokenizer(
        [question.lstrip() for question in examples["question"]], examples["context"],
        truncation="only_second", max_length=max_length, stride=stride,
        return_overflowing_tokens=True, return_offsets_mapping=True, padding="max_length",
    )
    mapping = encoded.pop("overflow_to_sample_mapping")
    starts, ends, ids = [], [], []
    for i, sample in enumerate(mapping):
        offsets = encoded["offset_mapping"][i]
        sequence = encoded.sequence_ids(i)
        context_tokens = [j for j, value in enumerate(sequence) if value == 1]
        if not context_tokens:
            raise ValueError("A tokenized QA window contains no context tokens")
        first, last = context_tokens[0], context_tokens[-1]
        answers = examples["answers"][sample]
        start = end = encoded["input_ids"][i].index(tokenizer.cls_token_id)
        if answers["answer_start"]:
            left = answers["answer_start"][0]
            right = left + len(answers["text"][0])
            if offsets[first][0] <= left and offsets[last][1] >= right:
                start, end = first, last
                while start <= last and offsets[start][0] <= left:
                    start += 1
                while end >= first and offsets[end][1] >= right:
                    end -= 1
                start, end = start - 1, end + 1
        starts.append(start)
        ends.append(end)
        ids.append(examples["id"][sample])
        encoded["offset_mapping"][i] = [offset if sequence[j] == 1 else None
                                        for j, offset in enumerate(offsets)]
    encoded["start_positions"], encoded["end_positions"] = starts, ends
    encoded["example_id"] = ids
    return dict(encoded)


def normalize_answer(text):
    text = "".join(character for character in text.lower() if character not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def answer_scores(prediction, reference):
    predicted, expected = normalize_answer(prediction), normalize_answer(reference)
    exact = float(predicted == expected)
    predicted, expected = predicted.split(), expected.split()
    if not predicted or not expected:
        return exact, float(predicted == expected)
    common = sum((Counter(predicted) & Counter(expected)).values())
    return exact, 2 * common / (len(predicted) + len(expected))


def score_predictions(examples, features, predictions, max_answer_length=30):
    """Combine all windows before scoring each original question once."""
    references = {example["id"]: example for example in examples}
    best = {}
    seen = set()
    for index, starts, ends in predictions:
        if index in seen:
            raise ValueError("Duplicate validation feature prediction")
        seen.add(index)
        feature = features[index]
        identifier = feature["example_id"]
        context = references[identifier]["context"]
        offsets = feature["offset_mapping"]
        start_indices = sorted(range(len(starts)), key=starts.__getitem__, reverse=True)[:20]
        end_indices = sorted(range(len(ends)), key=ends.__getitem__, reverse=True)[:20]
        for start in start_indices:
            for end in end_indices:
                if (end < start or end - start + 1 > max_answer_length or
                        offsets[start] is None or offsets[end] is None):
                    continue
                score = starts[start] + ends[end]
                if identifier not in best or score > best[identifier][0]:
                    best[identifier] = (score, context[offsets[start][0]:offsets[end][1]])
    if seen != set(range(len(features))):
        raise ValueError("Missing validation feature predictions")
    exact = f1 = 0.0
    answers = {}
    for identifier, example in references.items():
        prediction = best.get(identifier, (0, ""))[1]
        answers[identifier] = prediction
        scores = [answer_scores(prediction, text) for text in example["answers"]["text"] or [""]]
        exact += max(score[0] for score in scores)
        f1 += max(score[1] for score in scores)
    count = len(references)
    if not count:
        raise ValueError("Validation requires at least one question")
    return {"exact_match": 100 * exact / count, "f1": 100 * f1 / count,
            "validation_examples": count, "validation_features": len(features)}, answers
