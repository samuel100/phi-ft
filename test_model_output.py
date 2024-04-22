import onnxruntime_genai as og

prompt = "Cricket is a great game!"

model=og.Model(f'models/qlora/qlora-onnx_conversion-genai_export/gpu-cuda_model')

tokenizer = og.Tokenizer(model)

tokens = tokenizer.encode(prompt)

params=og.GeneratorParams(model)
params.set_search_options({"max_length":200})
params.input_ids = tokens

output_tokens=model.generate(params)[0]

text = tokenizer.decode(output_tokens)

print(text)
