from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import torch

def translate_file(input_file_path, output_file_path, src_lang="en_XX", target_lang="zh_CN"):
    # Load model and tokenizer
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

    # Ensure correct setting of the source language for the tokenizer
    tokenizer.src_lang = src_lang

    # Read the file content
    try:
        with open(input_file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
    except FileNotFoundError:
        print(f"Error: The file {input_file_path} does not exist.")
        return

    # Prepare the output file and process translations
    with open(output_file_path, 'w', encoding='utf-8') as file:
        for line in lines:
            # Strip whitespace to check if line is empty or just whitespace
            stripped_line = line.strip()
            if stripped_line:
                # Tokenize and encode the line for translation
                encoded = tokenizer(stripped_line, return_tensors="pt")
                # Generate translation with the specified target language
                generated_tokens = model.generate(
                    **encoded,
                    forced_bos_token_id=tokenizer.lang_code_to_id[target_lang]
                )
                # Decode the generated tokens to get the translated text
                translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

                # Write the original and translation to the output file
                file.write(f"Original: {stripped_line}\n")
                file.write(f"Translated: {translation}\n\n")

                # Print original and translated text to the console
                print(f"Original: {stripped_line}")
                print(f"Translated: {translation}\n")
            else:
                # Write a newline for empty lines in the input file
                file.write('\n')

    print(f"Translation completed. File saved to {output_file_path}\n")


