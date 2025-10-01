from evaluate import load
from model_utils import model_predict, get_p_true

squad_metric = load("squad_v2")


def construct_few_shot_prompt(
        *, model, tokenizer, dataset, indices, prompt, brief, brief_always, make_prompt,
        num_generations, metric, max_new_tokens, token_limit=2048):
    """
        Construct few shot prompt for p_true uncertainty metric.
    """

    # Call model n_shots many times
    few_shot_prompt = []
    all_responses = dict()

    for it, i in enumerate(indices):
        prompt_candidate = []
        example = dataset[i]
        question = example["question"]
        context = None

        if it != 0:
            prompt_candidate += ['\n']
        prompt_candidate += ['Question: ' + question]
        prompt_candidate += ['\nBrainstormed Answers: ']
        current_question = make_prompt(context, question, None, brief, brief_always)
        local_prompt = prompt + current_question

        responses = []
        for j in range(num_generations + 1):

            if j == 0:
                temperature = 0.1
            else:
                temperature = 1.0

            #response, _, _ = model.predict(local_prompt, temperature)
            response, _, _ = model_predict(model, tokenizer, local_prompt, temperature, max_new_tokens)

            responses.append(response)
            prompt_candidate += [f'{response.strip()} \n']
            if j == 0:
                # Save most likely response and compute correctness metric for it.
                most_likely_response = response
                is_correct = metric(response, example, model)
                answers = [answer for answer in example['answers']['text']]

        all_responses[i] = dict(
            responses=responses, most_likely_response=most_likely_response,
            is_correct=is_correct)

        prompt_candidate += ['Possible answer: ' + most_likely_response + '\n']
        prompt_candidate += ['Is the possible answer:\n']
        prompt_candidate += ['A) True\n']
        prompt_candidate += ['B) False\n']
        prompt_candidate += ['The possible answer is:']
        prompt_candidate += [' A' if is_correct else ' B']

        prompt_len = len(tokenizer.encode(''.join(few_shot_prompt + prompt_candidate)))
        # At test time, get a maximum of `num_generations * model.token_limit` extra tokens
        # 200 buffer for question and 'Possible Answer'.
        max_input_len = prompt_len + num_generations * max_new_tokens + 200

        if max_input_len < token_limit:
            few_shot_prompt.extend(prompt_candidate)
        else:
            print('Cutting of p_true prompt at length %d.', it)
            break

    return ''.join(few_shot_prompt), all_responses, it


def calculate_p_true(
        model, tokenizer, question, most_probable_answer, brainstormed_answers,
        few_shot_prompt, hint=False):
    """
        Calculate p_true uncertainty metric.
    """

    if few_shot_prompt:
        prompt = few_shot_prompt + '\n'
    else:
        prompt = ''

    prompt += 'Question: ' + question
    prompt += '\nBrainstormed Answers: '
    for answer in brainstormed_answers + [most_probable_answer]:
        prompt += answer.strip() + '\n'
    prompt += 'Possible answer: ' + most_probable_answer + '\n'
    if not hint:
        prompt += 'Is the possible answer:\n'
        prompt += 'A) True\n'
        prompt += 'B) False\n'
        prompt += 'The possible answer is:'
    else:
        prompt += 'Do the brainstormed answers match the possible answer? Respond with A if they do, if they do not respond with B. Answer:'

    log_prob = get_p_true(model, tokenizer, prompt)

    return log_prob