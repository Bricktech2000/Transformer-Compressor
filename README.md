# Transformer Compressor

_A high-efficiency text compressor leveraging large language models_

## Overview

This text compressor is a proof-of-concept for the use of transformer models in data compression algorithms. The input is tokenized then compressed using Huffman coding, with a unique Huffman tree for every input token, generated using the probabilities output by a large language model. This proof-of-concept uses Meta AI's [LLaMA](https://github.com/facebookresearch/llama) through [Lit-LLaMA](https://github.com/Lightning-AI/lit-llama) with pre-trained weights from [OpenLLaMA 7B](https://github.com/openlm-research/open_llama).

The goal of this project is to achieve high compression ratios while completely disregarding performance. The 413‑byte file [sample/sample.txt](sample/sample.txt) was compressed to 8% of its original size, a compression ratio of 12.91x, in a bit less than 18 hours.

Compression and decompression pseudo-code is as follows:

```python
def compress(input):
  tokens = tokenizer.encode(input)
  for prior_tokens, current_token in scan(tokens):
    probabilities = llm.predict_next_token(prior_tokens)
    huffman_tree = build_huffman_tree(probabilities)
    huffman_encoding = huffman_tree.encode(current_token)
    yield huffman_encoding

def decompress(input):
  prior_tokens = []
  while input:
    probabilities = llm.predict_next_token(prior_tokens)
    hunffman_tree = build_huffman_tree(probabilities)
    current_token = huffman_tree.decode(input)
    prior_tokens.append(current_token)
    yield tokenizer.decode(current_token)
```

## Installation

```bash
git clone https://github.com/Bricktech2000/Transformer-Compressor.git
cd Transformer-Compressor
git submodule update --init --recursive
pip3 install -r lit-llama/requirements.txt
python3 lit-llama/scripts/download.py --repo_id openlm-research/open_llama_7b --local_dir lit-llama/checkpoints/open-llama/7B
python3 lit-llama/scripts/convert_hf_checkpoint.py --checkpoint_dir lit-llama/checkpoints/open-llama/7B --output_dir lit-llama/checkpoints/lit-llama/7B --model_size 7B
python3 lit-llama/quantize/gptq.py --checkpoint_path lit-llama/checkpoints/lit-llama/7B/lit-llama.pth --tokenizer_path lit-llama/checkpoints/lit-llama/tokenizer.model --output_path lit-llama/checkpoints/lit-llama/7B/llama-gptq.4bit.pth --dtype bfloat16 --quantize gptq.int4
rm -r lit-llama/checkpoints/open-llama/
```

## Usage

```bash
# Compression
python3 trc.py compress <input_path> <output_path>
# Decompression
python3 trc.py decompress <input_path> <output_path>

# Sample Compression
python3 trc.py compress sample/sample.txt sample/compressed.trc
# Sample Decompression
python3 trc.py decompress sample/compressed.trc sample/decompressed.txt
```
