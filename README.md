# Olive FT + GenAI Example (Phi2)

This example shows how to use Olive to Finetune and Output an ONNX model that is compatible with the GenAI API for ONNX runtime. The Olive pipeline does a couple of passes:

1. QLoRA finetune.
2. ONNX conversion.
3. GenAI meta-data output.

The model output should be consumed using:

```Python
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
```

See [test_model_output.py](./test_model_output.py).