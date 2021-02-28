import torch

from allennlp import data
from allennlp.data import fields
from allennlp import modules


def check_nan_grads(words):
    "Encode a list of words, take a gradient, and check for NaN's."
    print(f"Checking {words}.")
    # Create indexer and embedder.
    tok_indexers = {"bert": data.token_indexers.PretrainedTransformerMismatchedIndexer(
        "bert-base-cased")}
    token_embedder = modules.token_embedders.PretrainedTransformerMismatchedEmbedder(
        "bert-base-cased")
    embedder = modules.text_field_embedders.BasicTextFieldEmbedder({"bert": token_embedder})

    # Convert words to tensor dict.
    vocab = data.Vocabulary()
    text_field = fields.TextField(
        [data.Token(word) for word in words], tok_indexers)
    text_field.index(vocab)
    token_tensor = text_field.as_tensor(text_field.get_padding_lengths())
    tensor_dict = text_field.batch_tensors([token_tensor])

    # Run forward pass. We need a scalar to take the gradient of, so just take the mean of the
    # embeddings.
    output = embedder(tensor_dict)
    loss = output.mean()
    loss.backward()

    # Check whether this produces an NaN in the model parameters.
    for name, param in embedder.named_parameters():
        grad = param.grad
        if grad is not None and torch.any(torch.isnan(param.grad)):
            print("Found NaN grad.")
            print("nan")
            print("Offending tensor_dict:")
            print(tensor_dict)
            print()
            return

    print()


####################

# This works fine.
example_safe = ["An", "example"]
check_nan_grads(example_safe)

# This produces NaN grads because of the empty string.
example_bad_empty = ["An", "", "example"]
check_nan_grads(example_bad_empty)

# This produces NaN grads because there's a weird character the indexer doesn't know about.
weird_character = "\uf732\uf730\uf730\uf733"
print(f"Weird character: {weird_character}.")
example_bad_unicode = ["A", weird_character, "example"]
check_nan_grads(example_bad_unicode)
