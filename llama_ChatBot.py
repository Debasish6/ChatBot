# from transformers import LlamaForCausalLM, LlamaTokenizer
# import sentencepiece

# # Load the tokenizer and model
# tokenizer = LlamaTokenizer.from_pretrained('llama/3.1')
# model = LlamaForCausalLM.from_pretrained('llama/3.1')

# def generate_response(input_text):
#     inputs = tokenizer(input_text, return_tensors="pt")
#     outputs = model.generate(**inputs, max_length=150, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return response


# # Flask Part

# from flask import Flask, render_template, request

# app = Flask(__name__)

# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/get")
# def get_bot_response():
#     user_text = request.args.get('msg')
#     bot_response = generate_response(user_text)
#     return str(bot_response)

# if __name__ == "__main__":
#     app.run(debug=True)

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from huggingface_hub import login
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
import transformers
import torch
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'



def main():
    try:
        # Model and Tokenizer Configuration
        model_name = "meta-llama/Llama-3.1-8B-Instruct"
        local_model_dir = "./local-llama-3.1-8B"
        
        # Login to Hugging Face
        login(token='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')

        # Download and cache the model and tokenizer if not already available
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=local_model_dir)
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=local_model_dir)
        
        # Initialize the model with empty weights for disk offload
        with init_empty_weights():
            config = transformers.AutoConfig.from_pretrained(local_model_dir)
            model = AutoModelForCausalLM.from_config(config)

        # Create device map for disk offload
        device_map = infer_auto_device_map(model)

        # Dispatch the model using load_checkpoint_and_dispatch
        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=local_model_dir,
            device_map=device_map,
            offload_folder=r'c:/Users/edominer/Python Project/ChatBot/ChatBot_with_Database'
        )

        # Set up the text generation pipeline
        text_generation_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.float16,
            device_map=device_map
        )

        # Prompt for text generation
        prompt = 'I have tomatoes, basil and cheese at home. What can I cook for dinner?\n'

        # Generate sequences
        sequences = text_generation_pipeline(
            prompt,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            truncation=True,
            max_length=400,
        )

        # Print the results
        print(sequences)
        for seq in sequences:
            print(f"Result: {seq['generated_text']}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

