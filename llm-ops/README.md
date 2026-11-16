# DSLM

Train a domain-specific language model, save the adapter, and query it with the same chat template the model was trained on.

The selected model is **[HuggingFaceTB/SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)**. It uses ChatML, and it is small enough to fine-tune on a CPU. A second config targets **Qwen2.5-0.5B-Instruct**, which has the same visible chat shape and is a better fit when you have a GPU.

`c:\personal\AI-Innovation\DSLM.ipynb` was not in this workspace. This repo is the runnable form of that workflow: JSONL chat rows, the model chat template, LoRA supervised fine-tuning with loss only on assistant tokens, a saved adapter plus manifest, and a query path that renders the same template. `notebooks/DSLM.ipynb` walks the same steps.

The sample domain is Harborline Logistics policy (hub cutoffs, claims, COD, packing). Swap `data/sample/*.jsonl` for your own domain.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
cd studio && npm install && cd ..
```

`requirements.txt` installs the CPU build of PyTorch. On a CUDA machine, install the CUDA wheel for torch before the rest of the file.

## Chat template of the selected model

SmolLM2-135M-Instruct. One training row renders as:

```
<|im_start|>system
You are Harborline, the policy assistant for Harborline Logistics. Answer only from Harborline policy. If a rule is not in the policy, say so.<|im_end|>
<|im_start|>user
What is the cutoff for same-day dispatch from the Pune hub?<|im_end|>
<|im_start|>assistant
Same-day dispatch from the Pune hub closes at 14:30 IST. Orders released after 14:30 IST leave on the next business day.<|im_end|>
```

Generation continues from the prefix that ends in `<|im_start|>assistant\n`. If a row has no system message, SmolLM2 inserts its own default system line (`You are a helpful AI assistant named SmolLM...`).

The JSONL row that produces that string:

```json
{"messages": [
  {"role": "system", "content": "You are Harborline, the policy assistant for Harborline Logistics. Answer only from Harborline policy. If a rule is not in the policy, say so."},
  {"role": "user", "content": "What is the cutoff for same-day dispatch from the Pune hub?"},
  {"role": "assistant", "content": "Same-day dispatch from the Pune hub closes at 14:30 IST. Orders released after 14:30 IST leave on the next business day."}
]}
```

`instruction` / `input` / `output` rows are accepted as well (`data/sample/instruction.jsonl`). The last message must be the assistant answer.

Qwen2.5-0.5B-Instruct renders the same ChatML text when you supply a system message. With no system message it inserts `You are Qwen, created by Alibaba Cloud. You are a helpful assistant.` Llama 3.2 uses a different header format; see `templates/llama-3.2-1b-instruct.sample.txt`.

Print the sample any time:

```bash
dslm template --config configs/smollm2-135m-instruct.yaml
dslm template --config configs/smollm2-135m-instruct.yaml --live
```

`--live` loads the tokenizer and uses the template stored on it.

## Train and save

```bash
dslm validate --config configs/smollm2-135m-instruct.yaml
dslm train --config configs/smollm2-135m-instruct.yaml
```

A saved run looks like this:

```
runs/smollm2-harborline/
  adapter/                  # LoRA weights + tokenizer
  manifest.json             # base model, loss, the rendered chat sample
  chat_template_sample.txt
  checkpoints/
```

`configs/smollm2-135m-instruct.smoke.yaml` runs two optimizer steps so you can confirm the save path on CPU. Those two steps do not teach the policy.

`runs/smollm2-harborline` is already a fit of the 12 sample rows (25 epochs, learning rate `1e-3`, LoRA rank 16, about a minute on CPU). Query that directory to see the policy answers. The same settings are in `configs/smollm2-135m-instruct.yaml`. For a larger corpus, lower the learning rate toward `2e-4` and train fewer passes.

On a GPU you can set `model.torch_dtype: bfloat16` and, if bitsandbytes is installed, `model.load_in_4bit: true`. 4-bit loading is refused on CPU.

## Query

```bash
dslm query --config configs/smollm2-135m-instruct.yaml \
  --run runs/smollm2-harborline \
  --question "What is the cutoff for same-day dispatch from the Pune hub?" \
  --show-prompt
```

If `--run` is omitted and the config output directory has no manifest, the base model is used.

The query studio talks to the API:

```bash
dslm serve --config configs/smollm2-135m-instruct.yaml --port 43123
cd studio && npm run dev
```

Open [DSLM Studio](http://127.0.0.1:43124). The first question loads the weights.

## Tests

```bash
pytest
```

Template, JSONL, and label-mask tests do not download weights. Training does.
