import onnxruntime_genai as og

user_input = "Cricket is so much fun!"
prompt = f"### Text: {user_input}\n### The tone is:\n"

model=og.Model(f'models/qlora/qlora-onnx_conversion-genai_export/gpu-cuda_model')

tokenizer = og.Tokenizer(model)

tokens = tokenizer.encode(prompt)

params=og.GeneratorParams(model)
params.set_search_options({"max_length":500})
params.input_ids = tokens

output_tokens=model.generate(params)[0]

text = str(tokenizer.decode(output_tokens))

print(text)
