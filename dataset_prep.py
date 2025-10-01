import random
import hashlib

from datasets import load_dataset

BRIEF_PROMPTS = {
    'default': "Answer the following question as briefly as possible.\n",
    'chat': 'Answer the following question in a single brief but complete sentence.\n'}

md5hash = lambda s: str(int(hashlib.md5(s.encode('utf-8')).hexdigest(), 16))


def construct_few_shot_prompt_from_indices(dataset, example_indices, brief, brief_always, make_prompt):
    """
        Given a dataset and indices, construct a few-shot prompt.
        :param dataset:
        :param example_indices:
        :param brief:
        :param brief_always:
        :param make_prompt:

    """
    if not brief_always:
        prompt = brief
    else:
        prompt = ''

    for example_index in example_indices:
        example = dataset[example_index]
        if 'context' in example.keys():
            context = example["context"]
        else:
            context = None

        question = example["question"]
        answer = example["numerical_answer"]

        prompt = prompt + make_prompt(context, question, answer, brief, brief_always)

    return prompt


def get_make_prompt(prompt_type, use_context):
    if prompt_type == 'default':
        def make_prompt(context, question, answer, brief, brief_always):
            prompt = ''
            if brief_always:
                prompt += brief
            if use_context and (context is not None):
                prompt += f"Context: {context}\n"
            prompt += f"Question: {question}\n"
            if answer:
                prompt += f"Answer: {answer}\n\n"
            else:
                prompt += 'Answer:'
            return prompt
    else:
        raise ValueError

    return make_prompt


def split_dataset(dataset):
    """
        Get indices of answerable and unanswerable questions.
        :param dataset:

    """

    answerable_indices = [i for i, example in enumerate(dataset) if example['answer_length'] > 0]
    unanswerable_indices = [i for i, example in enumerate(dataset) if example['answer_length'] == 0]

    return answerable_indices, unanswerable_indices


def dataset_pre_process(example):
    example['answer_explanation'] = example['answer']

    answer_str = example['answer'].split('#### ')[-1].strip().replace(',', '')
    int_answer = int(answer_str)
    fl_answer = float(answer_str)

    example['id'] = md5hash(str(example['question']))
    example['answer_length'] = len(answer_str)

    if int_answer == fl_answer:
        example['answer_numerical_type'] = 'int'
        example['numerical_answer'] = int_answer
    else:
        example['answer_numerical_type'] = 'float'
        example['numerical_answer'] = fl_answer

    return example


def load_ds(dataset_name):
    """
        Load dataset

    """
    train_dt, validation_dt = None, None
    if dataset_name == "gsm8k":
        dataset = load_dataset("openai/gsm8k", "main")
        dataset = dataset.map(dataset_pre_process)

        dataset = dataset.filter(lambda x: x['id'] != '229204758988173734073362288035613840709')

        test_valid_split = dataset['test'].train_test_split(test_size=0.2)
        train_dt = dataset["train"]
        validation_dt = test_valid_split["test"]

    return train_dt, validation_dt


def get_dataset(dataset_name, experiment_details, num_few_shot, running_parameters):
    """

    :param dataset_name:
    :param experiment_details:
    :param num_few_shot:
    :param running_parameters: brief_prompt, brief_always, enable_brief, use_context
    :return:
    """

    train_dataset, validation_dataset = load_ds(dataset_name)
    answerable_indices, unanswerable_indices = split_dataset(train_dataset)

    prompt_indices = random.sample(answerable_indices, num_few_shot)
    experiment_details['prompt_indices'] = prompt_indices
    remaining_answerable = list(set(answerable_indices) - set(prompt_indices))

    BRIEF = BRIEF_PROMPTS[running_parameters['brief_prompt']]
    arg = running_parameters['brief_always'] if running_parameters['enable_brief'] else True
    make_prompt = get_make_prompt(prompt_type='default', use_context=running_parameters['use_context'])
    prompt = construct_few_shot_prompt_from_indices(train_dataset, prompt_indices, BRIEF, arg, make_prompt)

    experiment_details['BRIEF'] = BRIEF
    experiment_details['prompt'] = prompt

    return train_dataset, validation_dataset, answerable_indices, unanswerable_indices, remaining_answerable, experiment_details

## Testing Code
# experiment_details = {}
# running_parameters = {'brief_prompt': 'default', 'brief_always': False, 'enable_brief': True, 'use_context': False}
# tr_dt, vl_dt, ans_ind, unans_ind, rm_answ, exp = get_dataset('gsm8k', experiment_details, 5, running_parameters)