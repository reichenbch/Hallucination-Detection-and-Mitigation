import gc
import os
import torch
import random
import logging
import numpy as np
from tqdm import tqdm
from model_utils import model_init, model_predict
from utils import get_metric, get_reference, save
from dataset_prep import get_dataset, get_make_prompt, split_dataset

import p_true as p_true_utils
from compute_uncertainity_measures import main as main_compute


def main(args):
    experiment_details = {'args': args}
    random.seed(10)

    # todo - clean this up
    metric = get_metric('squad')

    dataset_name = 'gsm8k'
    answerable_only = False
    compute_p_true = True
    brief_always = False
    enable_brief = True
    p_true_num_fewshot = 20
    num_generations = 10
    get_training_set_generations = True
    num_samples = 400
    get_training_set_generations_most_likely_only = True
    temperature = 1.0
    max_new_tokens = 50
    compute_accuracy_at_all_temps = True
    p_true_hint = False

    running_parameters = {'brief_prompt': 'default', 'brief_always': False,
                          'enable_brief': True, 'use_context': False}

    train_dataset, validation_dataset, answerable_indices, unanswerable_indices, remaining_answerable, experiment_details = get_dataset(
        dataset_name, experiment_details, 5, running_parameters)

    if answerable_only:
        unanswerable_indices = []
        val_answerable, val_unanswerable = split_dataset(validation_dataset)
        del val_unanswerable
        validation_dataset = [validation_dataset[i] for i in val_answerable]

    # Initialize model.
    model_name = 'Mistral-7B-v0.1'
    tokenizer, model = model_init(model_name)
    make_prompt = get_make_prompt(prompt_type='default', use_context=running_parameters['use_context'])

    # Initialize prompt for p_true baseline.
    if compute_p_true:
        p_true_indices = random.sample(answerable_indices, p_true_num_fewshot)

        remaining_answerable = list(set(remaining_answerable) - set(p_true_indices))
        make_prompt = get_make_prompt(prompt_type='default', use_context=running_parameters['use_context'])

        # todo
        p_true_few_shot_prompt, p_true_responses, len_p_true = p_true_utils.construct_few_shot_prompt(
            model=model, tokenizer=tokenizer, dataset=train_dataset, indices=p_true_indices,
            prompt=experiment_details['prompt'], brief=experiment_details['BRIEF'],
            brief_always=brief_always and enable_brief,
            make_prompt=make_prompt, num_generations=num_generations,
            metric=metric, max_new_tokens=max_new_tokens)

        experiment_details['p_true_indices'] = p_true_indices
        experiment_details['p_true_responses'] = p_true_responses
        experiment_details['p_true_few_shot_prompt'] = p_true_few_shot_prompt

    # Start answer generation.
    for dataset_split in ['train', 'validation']:

        # This will store all input data and model predictions.
        accuracies, generations, results_dict, p_trues = [], {}, {}, []

        if dataset_split == 'train':
            if not get_training_set_generations:
                continue
            dataset = train_dataset
            possible_indices = list(set(remaining_answerable) | set(unanswerable_indices))

        else:
            dataset = validation_dataset
            possible_indices = range(0, len(dataset))

        # Evaluate over random subset of the datasets.
        indices = random.sample(possible_indices, min(num_samples, len(dataset)))
        experiment_details[dataset_split] = {'indices': indices}

        if num_samples > len(dataset):
            print('Not enough samples in dataset. Using all %d samples.', len(dataset))

        it = 0
        for index in tqdm(indices):
            if (it + 1 % 10) == 0:
                gc.collect()
                torch.cuda.empty_cache()
            it += 1

            # Grab example at index.
            example = dataset[index]
            question, context = example["question"], None
            generations[example['id']] = {'question': question, 'context': context}
            correct_answer = example['numerical_answer']

            current_input = make_prompt(
                context, question, None, experiment_details['BRIEF'], brief_always and enable_brief)
            local_prompt = experiment_details['prompt'] + current_input

            full_responses = []

            # We sample 1 low temperature answer on which we will compute the
            # accuracy and num_generation high temperature answers which will
            # be used to estimate the entropy.

            if dataset_split == 'train' and get_training_set_generations_most_likely_only:
                num_generations = 1
            else:
                num_generations = num_generations + 1

            for i in range(num_generations):

                # Temperature for first generation is always `0.1`.
                temperature = 0.1 if i == 0 else temperature

                predicted_answer, token_log_likelihoods, (
                    embedding, emb_last_before_gen, emb_before_eos) = model_predict(model, tokenizer, local_prompt,
                                                                                    temperature, max_new_tokens,
                                                                                    return_latent=True)

                # Last token embedding
                embedding = embedding.cpu() if embedding is not None else None
                emb_last_before_gen = emb_last_before_gen.cpu() if emb_last_before_gen is not None else None
                emb_before_eos = emb_before_eos.cpu() if emb_before_eos is not None else None

                compute_acc = compute_accuracy_at_all_temps or (i == 0)
                if correct_answer and compute_acc:
                    acc = metric(predicted_answer, example, model)
                else:
                    acc = 0.0

                if i == 0:
                    accuracies.append(acc)
                    most_likely_answer_dict = {
                        'response': predicted_answer,
                        'token_log_likelihoods': token_log_likelihoods,
                        'embedding': embedding,
                        'accuracy': acc,
                        'emb_last_tok_before_gen': emb_last_before_gen,
                        'emb_tok_before_eos': emb_before_eos,
                    }

                    generations[example['id']].update({
                        'most_likely_answer': most_likely_answer_dict,
                        'reference': get_reference(example),
                    })
                else:
                    # Aggregate predictions over num_generations.
                    full_responses.append(
                        (predicted_answer, token_log_likelihoods, embedding, acc))

            # Append all predictions for this example to `generations`.
            generations[example['id']]['responses'] = full_responses

            if compute_p_true and dataset_split == 'validation':
                # Already compute p_true here. Avoid cost of generations in compute_uncertainty script.
                p_true = p_true_utils.calculate_p_true(
                    model, question, most_likely_answer_dict['response'],
                    [r[0] for r in full_responses], p_true_few_shot_prompt,
                    hint=p_true_hint)
                p_trues.append(p_true)

        # Save generations for that split.
        save(generations, f'{dataset_split}_generations.pkl')

        # Log overall accuracy.
        accuracy = np.mean(accuracies)
        print(f"Overall {dataset_split} split accuracy: {accuracy}")

        if dataset_split == 'validation':
            if compute_p_true:
                results_dict['uncertainty_measures'] = {
                    'p_false': [1 - p for p in p_trues],
                    'p_false_fixed': [1 - np.exp(p) for p in p_trues],
                }
            save(results_dict, 'uncertainty_measures.pkl')

    save(experiment_details, 'experiment_details.pkl')
    del model
    del tokenizer


if __name__ == '__main__':
    args = dict()
    args['temperature'] = 1.0
    args['num_samples'] = 400
    args['metric'] = 'squad'
    args['enable_brief'] = True

    args['p_true_hint'] = False
    args['enable_brief'] = True
    args['use_context'] = False
    args['max_new_tokens'] = 50

    args['brief_always'] = False
    args['brief_always'] = False
    args['num_generations'] = 10

    args['compute_p_true'] = True
    args['dataset_name'] = 'gsm8k'
    args['answerable_only'] = False
    args['p_true_num_fewshot'] = 20
    args['brief_prompt'] = 'default'

    args['compute_uncertainties'] = True
    args['get_training_set_generations'] = True
    args['compute_accuracy_at_all_temps'] = True
    args['get_training_set_generations_most_likely_only'] = True

    main(args)

    if args['compute_uncertainties']:
        main_compute(args)
