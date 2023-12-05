import transformers


def BuildTokenizer(model_name_or_path, cache_dir=None, model_max_length=512, **kwargs):
    # load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        model_max_length=model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )
    return tokenizer


def BuildAutoModel(model_name_or_path, cache_dir=None, num_labels=2, **kwargs):
    # load model
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        num_labels=num_labels,
        trust_remote_code=True,
    )
    return model


def dnabert_2_117m(cache_dir=None, num_labels=2, model_max_length=512, return_tokenizer=False, **kwargs):
    """DNA-BERT-2-117M from original paper (https://arxiv.org/abs/2306.15006).
        source https://github.com/Zhihan1996/DNABERT_2.
    """
    try:
        import accelerate, einops
    except ImportError:
        print('Please install "accelerate" and "einops" to use DNABERT-2')
    model_kwargs = dict(
        model_name_or_path='zhihan1996/DNABERT-2-117M',
        cache_dir='./saved_models/',
        num_labels=num_labels,
        model_max_length=model_max_length,
        **kwargs
    )
    model = BuildAutoModel(**model_kwargs)
    if return_tokenizer:
        tokenizer = BuildTokenizer(**model_kwargs)
        return model, tokenizer
    else:
        return model


if __name__ == '__main__':
    model, tokenizer = dnabert_2_117m(return_tokenizer=True)
    print(model)
