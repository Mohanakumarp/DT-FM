# Tamil summarisation with mT5

Use the configurable image with these runtime options:

```text
--model google/mt5-small
--dataset csebuetnlp/xlsum --dataset-config tamil
--source-column text --target-column summary
```

XL-Sum's article/summary JSONL files load without executing its legacy script.
Training uses `train`, validation uses `validation`, and the test split stays
held out. Default validation reports loss over up to ten batches per replica.
Assess generated Tamil summaries separately.

The new runner allocates mT5 blocks across pipeline stages and data replicas.
It preserves shared embedding, relative attention bias, and encoder-memory
gradients. Total ranks are not restricted to two or three.

- [Configurable models/datasets, image publishing, and checkpoints](./DISTRIBUTED_SUMMARIZATION.md)
- [Linux and Windows workers over Tailscale](./MT5_TAILSCALE.md)

Publish the updated runner once; workers then use only the image. Later model,
dataset, and laptop-count changes need no code ZIP or further rebuild.

An exported/assembled model is a normal Hugging Face directory:

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

path = "path/to/exported/model"
tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(path).eval()
inputs = tokenizer("உங்கள் தமிழ் செய்திக் கட்டுரையை இங்கே இடவும்.",
                   return_tensors="pt", truncation=True, max_length=512)
outputs = model.generate(**inputs, max_new_tokens=128, num_beams=4)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

Sources: [mT5](https://huggingface.co/google/mt5-small),
[XL-Sum](https://huggingface.co/datasets/csebuetnlp/xlsum).
