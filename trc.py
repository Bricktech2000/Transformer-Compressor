from typing import Optional
from pathlib import Path
import warnings
import heapq
import time
import sys
import os

import lightning as L
import torch

# load `lit_llama` from `lit-llama` submodule
wd = os.path.join(Path(__file__).parent.resolve(), 'lit-llama')  # noqa
sys.path.append(str(wd))  # noqa

from lit_llama.utils import lazy_load, llama_model_lookup, quantization
from lit_llama import LLaMA, Tokenizer


# taken from `lit-llama/generate.py` with `return probs` added
@torch.no_grad()
def generate(
    model: LLaMA,
    idx: torch.Tensor,
    max_new_tokens: int,
    *,
    max_seq_length: Optional[int] = None,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    eos_id: Optional[int] = None,
) -> torch.Tensor:
  '''Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.

  The implementation of this function is modified from A. Karpathy's nanoGPT.

  Args:
      model: The model to use.
      idx: Tensor of shape (T) with indices of the prompt sequence.
      max_new_tokens: The number of new tokens to generate.
      max_seq_length: The maximum sequence length allowed.
      temperature: Scales the predicted logits by 1 / temperature
      top_k: If specified, only sample among the tokens with the k highest probabilities
      eos_id: If specified, stop generating any more token once the <eos> token is triggered
  '''
  # create an empty tensor of the expected final shape and fill in the current tokens
  T = idx.size(0)
  T_new = T + max_new_tokens
  if max_seq_length is None:
    max_seq_length = min(T_new, model.config.block_size)

  device, dtype = idx.device, idx.dtype
  # create an empty tensor of the expected final shape and fill in the current tokens
  empty = torch.empty(T_new, dtype=dtype, device=device)
  empty[:T] = idx
  idx = empty
  input_pos = torch.arange(0, T, device=device)

  if idx.device.type == 'xla':
    import torch_xla.core.xla_model as xm

    xm.mark_step()

  # generate max_new_tokens tokens
  for _ in range(max_new_tokens):
    x = idx.index_select(0, input_pos).view(1, -1)

    # forward
    logits = model(x, max_seq_length, input_pos)
    logits = logits[0, -1] / temperature

    # optionally crop the logits to only the top k options
    if top_k is not None:
      v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
      logits = torch.where(logits < v[[-1]], -float('Inf'), logits)

    probs = torch.nn.functional.softmax(logits, dim=-1)
    idx_next = torch.multinomial(probs, num_samples=1).to(dtype=dtype)

    return probs

    # advance
    input_pos = input_pos[-1:] + 1

    if idx.device.type == 'xla':
      xm.mark_step()

    # concatenate the new generation
    idx = idx.index_copy(0, input_pos, idx_next)

    # if <eos> token is triggered, return the output (stop generation)
    if idx_next == eos_id:
      return idx[:input_pos]  # include the EOS token

  return idx


def bits_to_bytes(bits):
  bits += [0] * (-len(bits) % 8)
  return bytes(sum(byte[b] << 7 >> b for b in range(8)) for byte in zip(*(iter(bits),) * 8))


def bytes_to_bits(bytes):
  return [int(b & (1 << 7 >> i) != 0) for b in bytes for i in range(8)]


# taken partly from `lit-llama/generate.py`
def main():
  if len(sys.argv) != 4:
    print('Usage: python3 trc.py (compress|decompress) <input_path> <output_path>')
    sys.exit(1)

  checkpoint_path = Path('lit-llama/checkpoints/lit-llama/7B/lit-llama.pth')
  tokenizer_path = Path('lit-llama/checkpoints/lit-llama/tokenizer.model')
  assert checkpoint_path.is_file(), checkpoint_path
  assert tokenizer_path.is_file(), tokenizer_path
  operation = sys.argv[1]
  input_path = Path(sys.argv[2])
  output_path = Path(sys.argv[3])

  precision = 'bf16-true' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else '32-true'
  fabric = L.Fabric(devices=1, precision=precision)

  print('Loading model...')
  t0 = time.time()
  with lazy_load(checkpoint_path) as checkpoint:
    name = llama_model_lookup(checkpoint)

    with fabric.init_module(empty_init=True), quantization(mode=None):
      model = LLaMA.from_name(name)

    model.load_state_dict(checkpoint)
  print(f'Load time: {time.time() - t0:.02f} seconds')

  model.eval()
  model = fabric.setup(model)

  tokenizer = Tokenizer(tokenizer_path)

  L.seed_everything(0)

  with open(input_path, 'rb') as f:
    input = f.read()
  print(f'Input bytes: {input}')

  output = b''
  tokens = []
  t0 = time.perf_counter()
  if operation == 'compress':
    input_tokens = tokens = tokenizer.encode(input, bos=True, eos=False, device=fabric.device)
    print(f'Input tokens: {[tokenizer.decode(t).encode() for t in input_tokens]}')
    output_bits = []
    # start at 1 to ignore <bos>
    for x in range(1, input_tokens.size(0)):
      probabilities = generate(model, input_tokens[:x], max_new_tokens=1, temperature=1)
      probabilities = [(prob.item(), id, id) for id, prob in enumerate(probabilities)]
      # will sort by probability then by token id
      heapq.heapify(probabilities)

      path = []
      while len(probabilities) > 1:
        p1 = heapq.heappop(probabilities)
        p2 = heapq.heappop(probabilities)
        if p1[2] == input_tokens[x].item():
          path.append(0)
        if p2[2] == input_tokens[x].item():
          path.append(1)
        heapq.heappush(probabilities, (p1[0] + p2[0], p1[1], p1[2] if p1[2] == input_tokens[x].item() else p2[2]))
      output_bits.extend(reversed(path))

      print(f'Output bits: {output_bits}...')
      model.reset_cache()

    output_bits.append(1)  # mark end of file
    output = bits_to_bytes(output_bits)

  elif operation == 'decompress':
    input_bits = bytes_to_bits(input)
    while input_bits[-1] == 0:
      input_bits.pop()
    input_bits.pop()  # remove end of file marker
    print(f'Input bits: {input_bits}')
    output_tokens = torch.tensor([tokenizer.bos_id], device=fabric.device)
    while len(input_bits) > 0:
      probabilities = generate(model, output_tokens, max_new_tokens=1, temperature=1)
      probabilities = [(prob.item(), id, id) for id, prob in enumerate(probabilities)]
      # will sort by probability then by token id
      heapq.heapify(probabilities)

      while len(probabilities) > 1:
        print(probabilities[:20])
        p1 = heapq.heappop(probabilities)
        print(p1)
        p2 = heapq.heappop(probabilities)
        print(p2)
        # if two sums of probabilities collide, things will likely break
        # because the heap will attempt to sort by the pair (p1[2], p2[2]),
        # which differs from `compress`
        heapq.heappush(probabilities, (p1[0] + p2[0], p1[1], (p1[2], p2[2])))
      node = heapq.heappop(probabilities)[2]
      while isinstance(node, tuple):
        current_bit = input_bits.pop(0)
        if current_bit == 0:
          node = node[0]
        if current_bit == 1:
          node = node[1]
      output_tokens = tokens = torch.cat((output_tokens, torch.tensor([node], device=fabric.device)))

      print(f'Output tokens: {[tokenizer.decode(t).encode() for t in output_tokens]}...')
      model.reset_cache()

    output = tokenizer.decode(output_tokens).encode()

  else:
    print('Invalid operation')
    sys.exit(1)

  print(f'Output bytes: {output}')
  with open(output_path, 'wb+') as f:
    f.write(output)

  if fabric.device.type == 'cuda':
    print(f'Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB', file=sys.stderr)

  print(f'Done.')
  print(f'')
  print(f'Input size: {len(input) * 8} bits')
  print(f'Output size: {len(output) * 8} bits')
  print(f'Token count: {len(tokens)} tokens')
  print(f'Ratio: {max(len(input), len(output)) / min(len(input), len(output)):.02f}x')
  print(f'Input throughput: {len(input) * 8 / (time.perf_counter() - t0)} bits/second')
  print(f'Output throughput: {len(output) * 8 / (time.perf_counter() - t0)} bits/second')
  print(f'Token throughput: {len(tokens) / (time.perf_counter() - t0)} tokens/second')
  print(f'Total time: {time.perf_counter() - t0:.02f} seconds')


# taken mostly from `lit-llama/generate.py`
if __name__ == '__main__':
  torch.set_float32_matmul_precision('high')
  warnings.filterwarnings(
      # Triggered internally at ../aten/src/ATen/EmptyTensor.cpp:31
      'ignore',
      message='ComplexHalf support is experimental and many operators don\'t support it yet'
  )
  warnings.filterwarnings(
      # Triggered in bitsandbytes/autograd/_functions.py:298
      'ignore',
      message='MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization',
  )
  main()
