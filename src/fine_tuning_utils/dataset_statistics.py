MAX_TOKENS_PER_EXAMPLE = 16385

MIN_TARGET_EXAMPLES = 100
MAX_TARGET_EXAMPLES = 25000
MIN_DEFAULT_EPOCHS = 1
MAX_DEFAULT_EPOCHS = 25

def estimate_n_epochs(n_train_examples: int, n_epochs: int) -> int:
    """
    Estimate the default number of epochs based on the target data size constraints.
    """
    n_epochs = n_epochs
    if n_train_examples * n_epochs < MIN_TARGET_EXAMPLES:
        # Not enough total examples; increase epochs
        n_epochs = min(MAX_DEFAULT_EPOCHS, MIN_TARGET_EXAMPLES // n_train_examples)
    elif n_train_examples * n_epochs > MAX_TARGET_EXAMPLES:
        # Too many total examples; reduce epochs
        n_epochs = max(MIN_DEFAULT_EPOCHS, MAX_TARGET_EXAMPLES // n_train_examples)
    return n_epochs

def calculate_billing_tokens(convo_lens):
    """
    Calculate how many tokens will be billed for in the dataset.
    """
    n_billing_tokens_in_dataset = sum(min(MAX_TOKENS_PER_EXAMPLE, length) for length in convo_lens)
    return n_billing_tokens_in_dataset

def print_billing_info(n_epochs: int, n_billing_tokens_in_dataset: int, model: str):
    """
    Print the dataset billing info used in finetuning.
    """
    one_megatoken_price = {
        'gpt-4o': 25,
        'gpt-4o-mini': 3
    }
    print(f"Dataset has ~{n_billing_tokens_in_dataset} tokens that will be charged for during training")
    print(f"By default, you'll train for {n_epochs} epochs on this dataset")
    full_training_tokens = n_epochs * n_billing_tokens_in_dataset
    print(f"By default, you'll be charged for ~{full_training_tokens} tokens")
    print(f'Estimated cost: ${one_megatoken_price[model] * (full_training_tokens / 10**6)}')

