### Requirements
You can install the required packages with the following command:
```bash
cd latex2sympy
pip install -e .
cd ..
pip install -r requirements.txt 
pip install vllm==0.5.1 --no-build-isolation
pip install transformers==4.42.3
```

## Acknowledgement
The codebase is adapted from [Qwen2.5-Math](https://github.com/QwenLM/Qwen2.5-Math).
