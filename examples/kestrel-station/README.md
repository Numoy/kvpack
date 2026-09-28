# Example: Kestrel Station

`manual.md` is the operations manual of a research station under the Antarctic ice. It
is **invented**: no model has seen it during training, so any correct answer about it
must come from the cartridge (or from RAG or the prompt).

`questions.jsonl` has 28 questions. Each lists the facts a correct answer must contain,
so `kvpack eval` can grade answers without an LLM judge.

```bash
# build a cartridge (use a GPU and a larger model for better results)
kvpack build examples/kestrel-station/manual.md \
  --model Qwen/Qwen3-4B --tokens 256 --samples 1024 --out kestrel.safetensors

# compare it with RAG, the full manual and no context
kvpack eval kestrel.safetensors --docs examples/kestrel-station/manual.md \
  -q examples/kestrel-station/questions.jsonl -o eval.json

# or look at individual answers side by side
kvpack compare kestrel.safetensors --docs examples/kestrel-station/manual.md \
  -q "What is the name of the bell that is rung when a borehole reaches the lake?"
```

The results for Qwen3-0.6B on a laptop are in the main
[README](../../README.md#reproduce-the-benchmark).
