import os
import pickle
import random
import numpy as np
from copy import deepcopy
from model_utils import model_init
from utils import get_metric, save
from collections import defaultdict

# from analyze_results import analyze_run

from p_ik import get_p_ik
import p_true as p_true_utils
from dataset_prep import get_dataset, get_make_prompt
from semantic_entropy import (get_semantic_ids, logsumexp_by_id, predictive_entropy, predictive_entropy_rao,
                              cluster_assignment_entropy, context_entails_response, EntailmentDeberta)

EXP_DETAILS = 'experiment_details.pkl'


def main(args):
    compute_predictive_entropy = True
    compute_p_true_in_compute_stage = False
    entailment_model = 'deberta'
    use_all_generations = True
    reuse_entailment_model = False
    use_num_generations = -1
    max_new_tokens = 50
    recompute_accuracy = False
    compute_context_entails_response = False
    condition_on_question = True
    strict_entailment = True
    num_eval_samples = int(1e19)
    compute_p_ik = True
    compute_p_ik_answerable = False
    dataset_name = 'gsm8k'
    args_metric = 'squad_v2'

    if compute_predictive_entropy:
        print('Beginning loading for entailment model.')
        if entailment_model == 'deberta':
            entailment_model = EntailmentDeberta()
        else:
            raise ValueError

    def restore(filename):
        class Restored:
            name = f'{filename}'

        return Restored

    experiment_details = {'args': args}
    running_parameters = {'brief_prompt': 'default', 'brief_always': False,
                          'enable_brief': True, 'use_context': False}

    filename = 'train_generations.pkl'
    with open(f'{filename}', "rb") as infile:
        train_generations = pickle.load(infile)

    if compute_p_true_in_compute_stage:
        old_exp_file = restore(EXP_DETAILS)
        with open(old_exp_file.name, "rb") as infile:
            old_exp = pickle.load(infile)

        if reuse_entailment_model:
            pt_model = entailment_model.model
        else:
            model_name = 'Mistral-7B-v0.1'
            tokenizer, pt_model = model_init(model_name)

        pt_train_dataset, pt_validation_dataset, answerable_indices, unanswerable_indices, remaining_answerable, experiment_details = get_dataset(
            dataset_name, experiment_details, 5, running_parameters)

        # Reduce num generations used in p_true if needed!
        if not use_all_generations:
            if use_num_generations == -1:
                raise ValueError
            num_gen = use_num_generations
        else:
            num_gen = 10

        p_true_indices = random.sample(answerable_indices, 20)  # args.p_true_num_fewshot = 20

        p_true_few_shot_prompt, p_true_responses, len_p_true = p_true_utils.construct_few_shot_prompt(
            model=pt_model,
            tokenizer=tokenizer,
            dataset=pt_train_dataset,
            indices=p_true_indices,
            prompt=old_exp['prompt'],
            brief=old_exp['BRIEF'],
            brief_always=old_exp['args'].brief_always and old_exp['args'].enable_brief,
            make_prompt=get_make_prompt(old_exp['args'], use_context=running_parameters['use_context']),
            num_generations=num_gen,
            metric=get_metric(old_exp['args'].metric), max_new_tokens=max_new_tokens)

        del p_true_responses, pt_train_dataset

    if recompute_accuracy:
        print('Recompute accuracy enabled. This does not apply to precomputed p_true!')
        metric = get_metric(args_metric)

    result_dict_pickle = restore('uncertainty_measures.pkl')
    with open(result_dict_pickle.name, "rb") as infile:
        result_dict = pickle.load(infile)

    if 'semantic_ids' not in result_dict:
        result_dict['semantic_ids'] = []

    validation_generations_pickle = restore('validation_generations.pkl')
    with open(validation_generations_pickle.name, 'rb') as infile:
        validation_generations = pickle.load(infile)

    entropies, accuracies = defaultdict(list), defaultdict(list)
    validation_embeddings, validation_is_true, validation_answerable = [], [], []
    p_trues = []
    count = 0  # pylint: disable=invalid-name

    def is_answerable(generation):
        return len(generation['reference']['answers']['text']) > 0

    # Loop over datapoints and compute validation embeddings, accuracies and entropies.
    for idx, tid in enumerate(validation_generations):
        example = validation_generations[tid]
        question = example['question']
        context = example['context']
        full_responses = example["responses"]
        most_likely_answer = example['most_likely_answer']

        if not use_all_generations:
            if use_num_generations == -1:
                raise ValueError
            responses = [fr[0] for fr in full_responses[:args.use_num_generations]]
        else:
            responses = [fr[0] for fr in full_responses]

        if recompute_accuracy:
            if is_answerable(example):
                acc = metric(most_likely_answer['response'], example, None)
            else:
                acc = 0.0  # pylint: disable=invalid-name
            validation_is_true.append(acc)

        else:
            validation_is_true.append(most_likely_answer['accuracy'])

        validation_answerable.append(is_answerable(example))
        validation_embeddings.append(most_likely_answer['embedding'])

        if compute_predictive_entropy:
            # Token log likelihoods. Shape = (n_sample, n_tokens)
            if not use_all_generations:
                log_liks = [r[1] for r in full_responses[:use_num_generations]]
            else:
                log_liks = [r[1] for r in full_responses]

            for i in log_liks:
                assert i

            if compute_context_entails_response:
                # Compute context entails answer baseline.
                entropies['context_entails_response'].append(context_entails_response(
                    context, responses, entailment_model))

            if condition_on_question and entailment_model == 'deberta':
                responses = [f'{question} {r}' for r in responses]

            # Compute semantic ids.
            semantic_ids = get_semantic_ids(
                responses, model=entailment_model,
                strict_entailment=strict_entailment, example=example)

            result_dict['semantic_ids'].append(semantic_ids)

            # Compute entropy from frequencies of cluster assignments.
            entropies['cluster_assignment_entropy'].append(cluster_assignment_entropy(semantic_ids))

            # Compute entropies with and without length normalized token probabilities.
            for agg_name, agg_func in zip(['', '_sum'], [np.mean, np.sum]):
                log_liks_agg = [agg_func(log_lik) for log_lik in log_liks]

                # Compute standard entropy.
                entropies['regular_entropy' + agg_name].append(predictive_entropy(log_liks_agg))

                # Compute semantic entropies with summing and with averaging probabilities within the cluster.
                cluster_agg_names = ['', '_sum-normalized', '_sum-normalized-rao', '_cmean']
                cluster_aggs = ['sum', 'sum_normalized', 'sum_normalized', 'mean']
                for cluster_agg_name, cluster_agg in zip(cluster_agg_names, cluster_aggs):
                    log_likelihood_per_semantic_id = logsumexp_by_id(semantic_ids, log_liks_agg, agg=cluster_agg)
                    name = 'semantic_entropy' + agg_name + cluster_agg_name

                    if cluster_agg_name != '_sum-normalized-rao':
                        pe = predictive_entropy(log_likelihood_per_semantic_id)
                    else:
                        pe = predictive_entropy_rao(log_likelihood_per_semantic_id)

                    entropies[name].append(pe)

                    # For the semantic uncertainties, we can also change the prediction, by first selecting the semantic
                    # cluster with the highest probability, and then selecting the generation with the highest probability
                    # within that cluster.
                    # NOTE: nanargmax because we currently have some clusters with empty generations.
                    max_cluster_id = np.nanargmax(log_likelihood_per_semantic_id)
                    # Filter log_liks to max cluster.
                    generations_in_cluster = np.array(log_liks_agg)
                    generations_in_cluster[np.array(semantic_ids) != max_cluster_id] = -np.inf
                    # Select generation with new max probability.
                    max_idx_in_cluster = np.argmax(generations_in_cluster)
                    # Accuracies for alternative generations saved at last index.
                    accuracies[name].append(full_responses[max_idx_in_cluster][-1])

            # pylint: disable=invalid-name
            log_str = 'semantic_ids: %s, avg_token_log_likelihoods: %s, entropies: %s'
            entropies_fmt = ', '.join([f'{i}:{j[-1]:.2f}' for i, j in entropies.items()])

        if compute_p_true_in_compute_stage:
            p_true = p_true_utils.calculate_p_true(
                pt_model, tokenizer, question, most_likely_answer['response'],
                responses, p_true_few_shot_prompt,
                hint=old_exp['args'].p_true_hint)
            p_trues.append(p_true)
            print('p_true: %s', np.exp(p_true))

        count += 1
        if count >= num_eval_samples:
            print('Breaking out of main loop.')
            break

    print('Accuracy on original task: %f', np.mean(validation_is_true))
    validation_is_false = [1.0 - is_t for is_t in validation_is_true]
    result_dict['validation_is_false'] = validation_is_false

    validation_unanswerable = [1.0 - is_a for is_a in validation_answerable]
    result_dict['validation_unanswerable'] = validation_unanswerable
    print('Unanswerable prop on validation: %f', np.mean(validation_unanswerable))

    if 'uncertainty_measures' not in result_dict:
        result_dict['uncertainty_measures'] = dict()

    if compute_predictive_entropy:
        result_dict['uncertainty_measures'].update(entropies)
        accuracies_mean = {k: np.mean(v) for k, v in accuracies.items()}
        print('Accuracy on original task from cluster-based generations: %s', accuracies_mean)

        result_dict['alt_validation_accuracies_mean'] = accuracies_mean
        result_dict['alt_validation_is_false'] = {k: [1 - vi for vi in v] for k, v in accuracies.items()}

    if compute_p_ik or compute_p_ik_answerable:
        # Assemble training data for embedding classification.
        train_is_true, train_embeddings, train_answerable = [], [], []
        for tid in train_generations:
            most_likely_answer = train_generations[tid]['most_likely_answer']
            train_embeddings.append(most_likely_answer['embedding'])
            train_is_true.append(most_likely_answer['accuracy'])
            train_answerable.append(is_answerable(train_generations[tid]))
        train_is_false = [0.0 if is_t else 1.0 for is_t in train_is_true]
        train_unanswerable = [0.0 if is_t else 1.0 for is_t in train_answerable]
        print('Unanswerable prop on p_ik training: %f', np.mean(train_unanswerable))

    if compute_p_ik:
        print('Starting training p_ik on train embeddings.')
        # Train classifier of correct/incorrect.
        p_ik_predictions = get_p_ik(
            train_embeddings=train_embeddings, is_false=train_is_false,
            eval_embeddings=validation_embeddings, eval_is_false=validation_is_false)
        result_dict['uncertainty_measures']['p_ik'] = p_ik_predictions
        print('Finished training p_ik on train embeddings.')

    if compute_p_ik_answerable:
        # Train classifier of answerable/unanswerable:
        p_ik_predictions = get_p_ik(
            train_embeddings=train_embeddings, is_false=train_unanswerable,
            eval_embeddings=validation_embeddings, eval_is_false=validation_unanswerable)
        result_dict['uncertainty_measures']['p_ik_unanswerable'] = p_ik_predictions

    if compute_p_true_in_compute_stage:
        result_dict['uncertainty_measures']['p_false'] = [1 - p for p in p_trues]
        result_dict['uncertainty_measures']['p_false_fixed'] = [1 - np.exp(p) for p in p_trues]

    save(result_dict, 'uncertainty_measures.pkl')

    if compute_predictive_entropy:
        entailment_model.save_prediction_cache()

    # if args.analyze_run:
    #    analyze_run(wandb.run.id)


if __name__ == '__main__':
    args = dict()
    args['compute_uncertainties'] = True

    if args['compute_uncertainties']:
        args['max_new_tokens'] = 50
        args['use_num_generations'] = -1
        args['recompute_accuracy'] = False
        args['use_all_generations'] = True

        args['entailment_model'] = 'deberta'
        args['compute_uncertainties'] = False
        args['reuse_entailment_model'] = False
        args['compute_predictive_entropy'] = True

        args['compute_p_true_in_compute_stage'] = False
        args['compute_context_entails_response'] = False

        args['strict_entailment'] = True
        args['num_eval_samples'] = int(1e19)
        args['compute_p_ik'] = True
        args['compute_p_ik_answerable'] = False
        args['dataset_name'] = 'gsm8k'
        args['args_metric'] = 'squad_v2'

        print("Args: %s", args)
        main(args)
