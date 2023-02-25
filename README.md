<center>
<img src="./banner.svg"/>
</center>

## What is ClosedAPI?

This is a set of APIs that essentially wrap Huggingface's [transformers](https://github.com/huggingface/transformers) and [diffusers](https://github.com/huggingface/diffusers) libraries to mimic some APIs from some companies.

It is not to say that monetization, proprietary models or private datasets are bad, but rather to show that it is possible to do the same thing with open source tools with transparency and more control over your decisions.

## How to use it?

Don't use it. This is just a proof of concept.

But if you want to contribute to it to make it better, here are some tasks it can do, for now:

### Text Classification

```
from closedapi.tasks import classify
classify("EleutherAI/gpt-neo-1.3B", [
    "how are you",
    ], [
    ("hello", "A"),
    ("goodbye", "B"),
    ("what's up", "A"),
    ("see you later", "B"),
])
# ["A"]
```

### Text Generation or Completion

```
from closedapi.tasks import completion
completion("gpt2", "Hello world")
# [{'generated_text': ' that\'s what you want from your life, not my life."\n\nAs'}]
```

### Text Embedding

```
from closedapi.tasks import embed
results = detect_language("papluca/xlm-roberta-base-language-detection", ["Hello world!", "Bonjour le monde!"])
results[0]["label"]
# 'en'
results[1]["label"]
# 'fr'
```

### Language Detection

```
from closedapi.tasks import detect_language
detect_language("Hello world!")
# 'en'
```

### Image Generation
```
from closedapi.tasks import image_generate

for image in image_generate("a dog", n=3):
    image.show()
```

## TODO

- [ ] Fake but required API keys
- [ ] Separate model services from the APIs, maybe something like [transformer-deploy](https://github.com/ELS-RD/transformer-deploy)
- [ ] Full compatibility with the original APIs (edit, generation, etc.)
- [ ] Default models for each task
- [ ] Documentations
- [ ] Pypi package