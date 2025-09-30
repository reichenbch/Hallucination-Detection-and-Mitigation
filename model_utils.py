import torch
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

stop_sequences = ['\n\n\n\n', '\n\n\n', 'Question:', 'Context:']
# '\n\n', '\n',


class StoppingCriteriaSub(StoppingCriteria):
    """
        Stop generations when they match a particular text or token.
    """

    def __init__(self, stops, tokenizer, match_on='text', initial_length=None):
        super().__init__()
        self.stops = stops
        self.initial_length = initial_length
        self.tokenizer = tokenizer
        self.match_on = match_on
        if self.match_on == 'tokens':
            self.stops = [torch.tensor(self.tokenizer.encode(i))  # .to('cuda')
                          for i in self.stops]
            print(self.stops)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs):
        del scores
        for stop in self.stops:
            if self.match_on == 'text':
                generation = self.tokenizer.decode(input_ids[0][self.initial_length:], skip_special_tokens=False)
                match = stop in generation
            elif self.match_on == 'tokens':
                # Can be dangerous due to tokenizer ambiguities.
                match = stop in input_ids[0][-len(stop):]
            else:
                raise
            if match:
                return True
        return False


def model_init(model_name):
    """
        Model Initialization

        :param model_name:
        :return:
    """
    tokenizer, model = None, None
    if model_name == 'Mistral-7B-v0.1':
        torch.cuda.empty_cache()

        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", device_map="auto",
                                                  token_type_ids=None, clean_up_tokenization_spaces=False)
        model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", device_map="auto",
                                                     attn_implementation='eager', max_memory={0: '80GIB'}).to('cuda:0')

    return tokenizer, model


def model_predict(model, tokenizer, input_data, temperature, max_new_tokens):

    token_limit = 2048
    stop_seq = stop_sequences + [tokenizer.eos_token]

    inputs = tokenizer(input_data, return_tensors="pt").to("cuda:0")

    if 'token_type_ids' in inputs:  # HF models seems has changed.
        del inputs['token_type_ids']
        pad_token_id = tokenizer.eos_token_id
    else:
        pad_token_id = None

    if stop_seq is not None:
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(
            stops=stop_seq,
            initial_length=len(inputs['input_ids'][0]),
            tokenizer=tokenizer)])
    else:
        stopping_criteria = None

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            output_hidden_states=True,
            temperature=temperature,
            do_sample=True,
            stopping_criteria=stopping_criteria,
            pad_token_id=pad_token_id,
        )

    if len(outputs.sequences[0]) > token_limit:
        raise ValueError(
            'Generation exceeding token limit %d > %d', len(outputs.sequences[0]), token_limit)

    full_answer = tokenizer.decode(
        outputs.sequences[0], skip_special_tokens=True)

    if full_answer.startswith(input_data):
        input_data_offset = len(input_data)
    else:
        input_data_offset = 0

    # Remove input from answer.
    answer = full_answer[input_data_offset:]

    # Remove stop_words from answer.
    stop_at = len(answer)
    sliced_answer = answer

    if stop_seq is not None:
        for stop in stop_seq:
            if answer.endswith(stop):
                stop_at = len(answer) - len(stop)
                sliced_answer = answer[:stop_at]
                break
        if not all([stop not in sliced_answer for stop in stop_seq]):
            error_msg = 'Error: Stop words not removed successfully!'
            error_msg += f'Answer: >{answer}< '
            error_msg += f'Sliced Answer: >{sliced_answer}<'
            print(error_msg)

    # Remove whitespaces from answer (in particular from beginning.)
    sliced_answer = sliced_answer.strip()
    token_stop_index = tokenizer(full_answer[:input_data_offset + stop_at], return_tensors="pt")['input_ids'].shape[1]
    n_input_token = len(inputs['input_ids'][0])
    n_generated = token_stop_index - n_input_token

    return_latent = True

    if n_generated == 0:
        print('Only stop_words were generated. For likelihoods and embeddings, taking stop word instead.')
        n_generated = 1

    if 'decoder_hidden_states' in outputs.keys():
        hidden = outputs.decoder_hidden_states
    else:
        hidden = outputs.hidden_states

    if len(hidden) == 1:
        last_input = hidden[0]
    elif (n_generated - 1) >= len(hidden):
        # if access idx is larger/equal
        last_input = hidden[-1]
    else:
        last_input = hidden[n_generated - 1]

    last_layer = last_input[-1]
    last_token_embedding = last_layer[:, -1, :].cpu()

    if return_latent:
        # Stack second last token embeddings from all layers
        if len(hidden) == 1:
            sec_last_input = hidden[0]
        elif (n_generated - 2) >= len(hidden):
            sec_last_input = hidden[-2]
        else:
            sec_last_input = hidden[n_generated - 2]
        sec_last_token_embedding = torch.stack([layer[:, -1, :] for layer in sec_last_input]).cpu()

        # Get the last input token embeddings (before generated tokens)
        last_tok_bef_gen_input = hidden[0]
        last_tok_bef_gen_embedding = torch.stack([layer[:, -1, :] for layer in last_tok_bef_gen_input]).cpu()

    # Get log_likelihoods.
    transition_scores = model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)
    log_likelihoods = [score.item() for score in transition_scores[0]]

    if len(log_likelihoods) == 1:
        log_likelihoods = log_likelihoods
    else:
        log_likelihoods = log_likelihoods[:n_generated]

    if len(log_likelihoods) == 0:
        raise ValueError

    hidden_states = (last_token_embedding,)
    hidden_states += (None, None)

    if return_latent:
        hidden_states += (sec_last_token_embedding, last_tok_bef_gen_embedding)
    else:
        hidden_states += (None, None)

    return_values = (sliced_answer, log_likelihoods, hidden_states)

    return return_values


def calculate_perplexity(input_data, model, tokenizer):
    """

        :param input_data:
        :param model:
        :param tokenizer:
        :return:
    """

    tokenized_data = tokenizer(input_data, return_tensors='pt').to('cuda')['input_ids']

    with torch.no_grad():
        model_output_true = model(tokenized_data, labels=tokenized_data)

    perplexity = - model_output_true.loss.item()
    return perplexity