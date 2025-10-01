import pickle
from evaluate import load


def get_reference(example):
    if 'numerical_answer' not in example:
        example = example['answer_explanation']

    answers = example['numerical_answer']

    answer_starts = example['answer'].rfind(answers)
    if answer_starts == -1:
        answer_starts = []

    reference = {'answers': {'answer_start': [answer_starts], 'text': [str(example['numerical_answer'])]}, 'id': example['id']}
    return reference


def get_metric(metric):
    if metric == 'squad':

        squad_metric = load("squad_v2")

        def metric(response, example, *args, **kwargs):
            # Compatibility with recomputation.
            if 'id' in example:
                exid = example['id']
            elif 'id' in example['reference']:
                exid = example['reference']['id']
            else:
                raise ValueError

            prediction = {'prediction_text': response, 'no_answer_probability': 0.0, 'id': exid}

            results = squad_metric.compute(
                predictions=[prediction],
                references=[get_reference(example)])
            return 1.0 if (results['f1'] >= 50.0) else 0.0

    return metric


def save(object, file):
    with open(f'{file}', 'wb') as f:
        pickle.dump(object, f)