#!/usr/bin/env python3


def few_shot_pipeline(instruction, data, model, tokenizer, device, temp, n_token):
    # Constract prompt
    # Generate answer
    # Extract answer
    # Save to the file (не забыть удалить старый файл)
    pass


def generate_answer(prompt, model, tokenizer, device='cpu', temp=0.1, n_token=100):
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    model_input = encoded
    model_input = model_input.to(device)
    generated_ids = model.generate(
        **model_input, do_sample=True,
        max_new_tokens=n_token, 
        temperature=temp, 
        # top_k=50, 
        num_return_sequences=1
        )
    decoded = tokenizer.batch_decode(generated_ids)

    return decoded

