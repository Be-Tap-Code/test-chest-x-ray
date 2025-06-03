# Evaluation function using BioBART
import torch
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import numpy as np
import pandas as pd
import os

# Ensure NLTK resources are downloaded
import nltk
# Download required NLTK resources
print("Downloading NLTK punkt tokenizer...")
nltk.download('punkt', quiet=True)
print("Downloading NLTK wordnet...")
nltk.download('wordnet', quiet=True)

def calculate_sample_metrics(generated_text, reference_text):
    """Calculate all metrics for a single sample."""
    # Tokenize texts
    gen_tokens = word_tokenize(generated_text.lower())
    ref_tokens = word_tokenize(reference_text.lower())
    
    # BLEU scores
    smoothing = SmoothingFunction().method1
    bleu1 = corpus_bleu([[ref_tokens]], [gen_tokens], weights=(1.0, 0, 0, 0), smoothing_function=smoothing)
    bleu2 = corpus_bleu([[ref_tokens]], [gen_tokens], weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
    bleu3 = corpus_bleu([[ref_tokens]], [gen_tokens], weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
    bleu4 = corpus_bleu([[ref_tokens]], [gen_tokens], weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
    
    # METEOR score
    meteor = meteor_score([ref_tokens], gen_tokens)
    
    # ROUGE scores
    rouge1 = calculate_rouge_n(ref_tokens, gen_tokens, n=1)
    rouge2 = calculate_rouge_n(ref_tokens, gen_tokens, n=2)
    rouge3 = calculate_rouge_n(ref_tokens, gen_tokens, n=3)
    rouge4 = calculate_rouge_n(ref_tokens, gen_tokens, n=4)
    rougeL = calculate_rouge_l(ref_tokens, gen_tokens)
    
    return {
        'BLEU-1': bleu1,
        'BLEU-2': bleu2,
        'BLEU-3': bleu3,
        'BLEU-4': bleu4,
        'METEOR': meteor,
        'ROUGE-1': rouge1,
        'ROUGE-2': rouge2,
        'ROUGE-3': rouge3,
        'ROUGE-4': rouge4,
        'ROUGE-L': rougeL
    }

def evaluate_model(mlp, biobart_model, test_loader, biobart_tokenizer, device):
    """Evaluates the model on the test set using BLEU, METEOR, and ROUGE metrics."""
    
    # Ensure models are in evaluation mode
    mlp.eval()
    biobart_model.eval()
    
    # Initialize lists to store results
    all_metrics = []
    generated_texts = []
    reference_texts = []
    
    print("\nStarting evaluation on the test set...")
    # Disable gradient calculations
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc='Evaluating', leave=False)
        for batch_idx, (input_batch, label_batch) in enumerate(progress_bar):
            # Move data to the evaluation device
            input_batch = input_batch.to(device)
            
            try:
                # Generate predictions
                embedded_input = mlp(input_batch)
                embedded_input_gen = embedded_input.unsqueeze(1)
                
                outputs = biobart_model.generate(
                    inputs_embeds=embedded_input_gen,
                    max_length=153,
                    num_beams=4,
                    early_stopping=True,
                    pad_token_id=biobart_tokenizer.pad_token_id,
                    eos_token_id=biobart_tokenizer.eos_token_id,
                )
                
                # Decode texts
                batch_generated_texts = biobart_tokenizer.batch_decode(outputs, skip_special_tokens=True)
                batch_reference_texts = biobart_tokenizer.batch_decode(label_batch, skip_special_tokens=True)
                
                # Calculate metrics for each sample in the batch
                for gen_text, ref_text in zip(batch_generated_texts, batch_reference_texts):
                    gen_text = gen_text.strip()
                    ref_text = ref_text.strip()
                    
                    # Store texts
                    generated_texts.append(gen_text)
                    reference_texts.append(ref_text)
                    
                    # Calculate and store metrics
                    metrics = calculate_sample_metrics(gen_text, ref_text)
                    all_metrics.append(metrics)
                    
                    # Print sample results
                    print(f"\n--- Sample {len(generated_texts)} ---")
                    print(f"Generated: {gen_text}")
                    print(f"Reference: {ref_text}")
                    print("Metrics:")
                    for metric_name, value in metrics.items():
                        print(f"  {metric_name}: {value:.4f}")
                    print("-" * 50)
            
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\nCUDA OOM during evaluation. Skipping batch.")
                    torch.cuda.empty_cache()
                    continue
                else:
                    print(f"Runtime Error during evaluation: {str(e)}")
                    raise e
            
            # Clear GPU memory periodically
            if device == torch.device("cuda") and batch_idx % 20 == 0:
                torch.cuda.empty_cache()
        
        progress_bar.close()

    # Calculate and print final results
    if all_metrics:
        print("\n=== Final Results ===")
        final_metrics = {}
        for metric_name in all_metrics[0].keys():
            values = [m[metric_name] for m in all_metrics]
            mean_value = np.mean(values)
            std_value = np.std(values)
            final_metrics[metric_name] = {
                'mean': mean_value,
                'std': std_value
            }
            print(f"{metric_name}:")
            print(f"  Mean: {mean_value:.4f}")
            print(f"  Std:  {std_value:.4f}")
        
        # Save results to CSV
        results_df = pd.DataFrame(all_metrics)
        results_df['Generated_Text'] = generated_texts
        results_df['Reference_Text'] = reference_texts
        results_df.to_csv('evaluation_results.csv', index=False)
        print("\nDetailed results saved to 'evaluation_results.csv'")
    else:
        print("Evaluation failed: No metrics were calculated.")

# Helper function for ROUGE-N calculation
def calculate_rouge_n(ref_tokens, gen_tokens, n=1):
    """Calculates ROUGE-N F1 score based on n-gram overlap."""
    if len(ref_tokens) == 0 or len(gen_tokens) == 0:
        return 0.0
    
    # Create n-grams
    def get_ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    
    ref_ngrams = get_ngrams(ref_tokens, n)
    gen_ngrams = get_ngrams(gen_tokens, n)
    
    if not ref_ngrams or not gen_ngrams:
        return 0.0
    
    # Count matching n-grams
    ref_ngram_counts = {}
    for ngram in ref_ngrams:
        ref_ngram_counts[ngram] = ref_ngram_counts.get(ngram, 0) + 1
    
    gen_ngram_counts = {}
    for ngram in gen_ngrams:
        gen_ngram_counts[ngram] = gen_ngram_counts.get(ngram, 0) + 1
    
    # Count overlapping n-grams
    overlap_count = 0
    for ngram, count in gen_ngram_counts.items():
        overlap_count += min(count, ref_ngram_counts.get(ngram, 0))
    
    # Calculate precision and recall
    precision = overlap_count / len(gen_ngrams) if gen_ngrams else 0.0
    recall = overlap_count / len(ref_ngrams) if ref_ngrams else 0.0
    
    # Calculate F1 score
    if precision + recall == 0:
        return 0.0
    f1 = (2 * precision * recall) / (precision + recall)
    return f1
        
# Helper function for ROUGE-L calculation using Longest Common Subsequence (LCS)
def calculate_rouge_l(ref_tokens, gen_tokens):
    #print("ref_tokens: ", ref_tokens)
    print("gen_tokens: ", gen_tokens)
    """Calculates ROUGE-L F1 score based on LCS."""
    # Find the length of the Longest Common Subsequence
    m, n = len(ref_tokens), len(gen_tokens)
    if m == 0 or n == 0: return 0.0
    # Initialize DP table (LCS lengths)
    L = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i-1] == gen_tokens[j-1]:
                L[i][j] = L[i-1][j-1] + 1
            else:
                L[i][j] = max(L[i-1][j], L[i][j-1])
    lcs_len = L[m][n]
    
    # Calculate Precision, Recall, and F1 for ROUGE-L
    recall = lcs_len / m if m > 0 else 0.0
    precision = lcs_len / n if n > 0 else 0.0
    
    if recall + precision == 0:
        f1 = 0.0
    else:
        f1 = (2 * recall * precision) / (recall + precision)
    return f1

# --- Run Evaluation --- 
# Ensure the test_loader is defined and models are loaded (potentially best checkpoint)
if 'test_loader' in locals() and test_loader is not None:
    # Optional: Load the best model checkpoint before evaluating
    if os.path.exists(checkpoint_path):
        print(f"\nLoading best model from {checkpoint_path} for final evaluation...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        mlp.load_state_dict(checkpoint['mlp_state_dict'])
        biobart_model.load_state_dict(checkpoint['biobart_state_dict'])
        print(f"Best model (Epoch {checkpoint['epoch']+1}) loaded.")
    else:
        print(f"Warning: Checkpoint {checkpoint_path} not found. Evaluating with current model state.")
        
    evaluate_model(mlp, biobart_model, test_loader, biobart_tokenizer, device)
else:
    print("\nError: Test loader not defined. Cannot run evaluation.")
