def call_musicflamingo(text, wav_path, processor, model):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"{text}"},
                {"type": "audio", "path": f"{wav_path}"},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=1024)

    decoded_outputs = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(decoded_outputs)
    return decoded_outputs[0]
