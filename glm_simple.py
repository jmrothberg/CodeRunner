#!/usr/bin/env python3


from mlx_lm import load, generate

model, tokenizer = load("/Users/jonathanrothberg/MLX_Models/GLM-4.7-6bit")

while True:
    prompt = input("Prompt: ")
    if prompt.lower() in ['quit', 'exit']:
        break
    response = generate(model, tokenizer, prompt)
    print("Response:", response)
