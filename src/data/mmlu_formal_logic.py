from datasets import load_dataset
import pandas as pd
import os


def load_data(args, split='validation'):
    split = 'validation' if split == 'train' else 'test'
    
    # Handle cache directory with permission fallback
    cache_dir = None
    if hasattr(args, 'data_dir') and args.data_dir:
        try:
            os.makedirs(args.data_dir, exist_ok=True)
            if os.access(args.data_dir, os.W_OK):
                cache_dir = args.data_dir
        except (OSError, PermissionError):
            cache_dir = None
    
    dataset = load_dataset('cais/mmlu', 'formal_logic', cache_dir=cache_dir)[split]
    dataset = pd.DataFrame(dataset)
    
    # Apply data_size limit if specified
    if args.data_size > 0:
        dataset = dataset.head(args.data_size)
    
    questions, labels = [], []
    choices = "ABCD"
    template = '{}\n(A) {}\n(B) {}\n(C) {}\n(D) {}\n\n'
    for query, options, answer in zip(dataset['question'], dataset['choices'], dataset['answer']):
        if len(options) != 4 :
            continue
        question = template.format(query, options[0], options[1], options[2], options[3])
        label = f"({choices[int(answer)]})"
        questions.append(question)
        labels.append(label)

    return questions, labels