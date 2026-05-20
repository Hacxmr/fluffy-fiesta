from datasets import load_dataset
import pandas as pd
import os

def extract_answer(options):
    """Extract the correct answer from options"""
    if isinstance(options, list) and len(options) > 0:
        return options[0]
    return None

def load_data(args, split='test'):
    """
    Load MMLU Pro Formal Logic dataset
    
    Args:
        args: Arguments object with data_dir and data_size
        split: 'train' or 'test'
    
    Returns:
        questions, labels: Lists of questions and corresponding answers
    """
    cache_dir = None
    if hasattr(args, 'data_dir') and args.data_dir:
        try:
            os.makedirs(args.data_dir, exist_ok=True)
            if os.access(args.data_dir, os.W_OK):
                cache_dir = args.data_dir
        except (OSError, PermissionError):
            cache_dir = None
    
    try:
        # Load MMLU Pro dataset
        dataset = load_dataset('TIGER-Lab/MMLU-Pro', 'formal_logic', cache_dir=cache_dir)[split]
    except Exception as e:
        print(f"Error loading MMLU-Pro formal_logic dataset: {e}")
        print("Attempting to use alternative dataset loading...")
        # Fallback: try standard MMLU
        try:
            dataset = load_dataset('cais/mmlu', 'formal_logic', cache_dir=cache_dir)[split]
        except Exception as e2:
            print(f"Error loading MMLU formal_logic dataset: {e2}")
            raise
    
    dataset = pd.DataFrame(dataset)
    
    if split == 'train':
        dataset = dataset.sample(frac=1, random_state=0).reset_index(drop=True)
    else:
        dataset = dataset.sample(frac=1, random_state=0).reset_index(drop=True)
    
    # Limit to data_size if specified
    if hasattr(args, 'data_size') and args.data_size > 0:
        dataset = dataset.head(args.data_size)
    
    questions = []
    labels = []
    
    # Handle different dataset formats
    if 'question' in dataset.columns and 'answer' in dataset.columns:
        # Format: separate question and answer columns
        for question, answer in zip(dataset['question'], dataset['answer']):
            questions.append(str(question))
            labels.append(answer)
    
    elif 'question' in dataset.columns and 'choices' in dataset.columns and 'answer' in dataset.columns:
        # Format: question with multiple choice options
        for idx, row in dataset.iterrows():
            question_text = str(row['question'])
            choices = row['choices'] if isinstance(row['choices'], list) else []
            answer = row['answer']
            
            # Format question with options
            formatted_q = f"{question_text}\n"
            if choices:
                for i, choice in enumerate(choices):
                    formatted_q += f"({chr(65+i)}) {choice}\n"
            
            questions.append(formatted_q.strip())
            labels.append(answer)
    
    else:
        # Fallback: use whatever columns available
        if 'prompt' in dataset.columns:
            questions = dataset['prompt'].tolist()
        elif 'question' in dataset.columns:
            questions = dataset['question'].tolist()
        else:
            questions = dataset.iloc[:, 0].tolist()
        
        if 'answer' in dataset.columns:
            labels = dataset['answer'].tolist()
        elif 'label' in dataset.columns:
            labels = dataset['label'].tolist()
        else:
            labels = dataset.iloc[:, -1].tolist()
    
    return questions, labels
