# Evaluation function using BioBART
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings

import torch
import logging
import numpy as np
import random
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from transformers import CLIPModel, CLIPProcessor, BartTokenizer, BartForConditionalGeneration
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import pandas as pd

# Ensure NLTK resources are downloaded
import nltk
# Download required NLTK resources
print("Downloading NLTK punkt tokenizer...")
nltk.download('punkt', quiet=True)
print("Downloading NLTK wordnet...")
nltk.download('wordnet', quiet=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fix random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Define MLP class
class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.dropout1 = torch.nn.Dropout(0.1)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        self.layer_norm = torch.nn.LayerNorm(output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.layer_norm(x)
        return x

def load_models():
    # Load CLIP model and processor
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    feature_extractor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Load BioBART tokenizer and model
    biobart_model_name = "GanjinZero/biobart-base"
    biobart_tokenizer = BartTokenizer.from_pretrained(biobart_model_name)
    biobart_model = BartForConditionalGeneration.from_pretrained(biobart_model_name).to(device)

    # Initialize MLP
    mlp_input_dim = 1024  # Concatenated CLIP features (512 * 2)
    mlp_hidden_dim = 1024
    mlp_output_dim = biobart_model.config.d_model
    mlp = MLP(input_dim=mlp_input_dim, hidden_dim=mlp_hidden_dim, output_dim=mlp_output_dim).to(device)

    # Load the best model checkpoint
    checkpoint_path = '/kaggle/input/best-model-iux-ray/vit_biobart_best_model.pt'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        mlp.load_state_dict(checkpoint['mlp_state_dict'])
        biobart_model.load_state_dict(checkpoint['biobart_state_dict'])
        logger.info(f"Loaded model from {checkpoint_path}")
    else:
        logger.error(f"Checkpoint not found at {checkpoint_path}")
        return None, None, None, None, None

    return clip_model, feature_extractor, biobart_model, biobart_tokenizer, mlp

def load_image(img_name, base_path="/kaggle/input/best-model-iux-ray/xray_images"):
    filename = os.path.basename(img_name.strip())
    full_path = os.path.join(base_path, filename)
    try:
        image = Image.open(full_path)
        image = image.resize((224, 224))
        image = np.asarray(image.convert("RGB"))
        image = cv2.resize(image, (224, 224))
        return image
    except FileNotFoundError:
        logger.warning(f"Image not found: {full_path}")
        return None

def extract_features(image1, image2, clip_model, feature_extractor):
    def extract_single_feature(image):
        if image is None:
            return None
        inputs = feature_extractor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = clip_model.get_image_features(**inputs)
        return outputs.squeeze(0)

    img1_features = extract_single_feature(image1)
    img2_features = extract_single_feature(image2)
    
    if img1_features is not None and img2_features is not None:
        combined_feature = np.concatenate([img1_features.cpu().numpy(), img2_features.cpu().numpy()], axis=-1)
        return combined_feature
    return None

def generate_report(feature_tensor, mlp, biobart_model, biobart_tokenizer):
    with torch.no_grad():
        embedded_input = mlp(feature_tensor)
        sample_embed_gen = embedded_input.unsqueeze(1)
        gen_ids = biobart_model.generate(
            inputs_embeds=sample_embed_gen,
            max_length=100,
            num_beams=4,
            early_stopping=True,
            pad_token_id=biobart_tokenizer.pad_token_id,
            eos_token_id=biobart_tokenizer.eos_token_id,
            do_sample=False
        )
        generated_text = biobart_tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()
    return generated_text

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

def evaluate_test_set():
    """Evaluate the model on the entire test set."""
    # Load models
    clip_model, feature_extractor, biobart_model, biobart_tokenizer, mlp = load_models()
    if None in [clip_model, feature_extractor, biobart_model, biobart_tokenizer, mlp]:
        logger.error("Failed to load models. Cannot proceed with evaluation.")
        return

    # Load test data
    test_dataset = pd.read_csv('/kaggle/input/data-split-csv/Test_Data.csv')
    
    # Initialize lists to store results
    all_metrics = []
    generated_texts = []
    reference_texts = []
    person_ids = []
    
    print("\nStarting evaluation on the test set...")
    for idx, row in tqdm(test_dataset.iterrows(), total=len(test_dataset), desc="Processing samples"):
        # Load images
        image1 = load_image(row['Image1'])
        image2 = load_image(row['Image2'])
        
        if image1 is not None and image2 is not None:
            # Extract features
            features = extract_features(image1, image2, clip_model, feature_extractor)
            if features is not None:
                # Normalize features
                mean, std = features.mean(), features.std()
                std = std if std > 1e-6 else 1.0
                features = (features - mean) / std
                
                # Convert to tensor
                feature_tensor = torch.tensor(features).float().unsqueeze(0).to(device)
                
                # Generate report
                generated_text = generate_report(feature_tensor, mlp, biobart_model, biobart_tokenizer)
                reference_text = row['Report'].strip()
                
                # Calculate metrics
                metrics = calculate_sample_metrics(generated_text, reference_text)
                
                # Store results
                all_metrics.append(metrics)
                generated_texts.append(generated_text)
                reference_texts.append(reference_text)
                person_ids.append(row['Person_id'])
                
                # Print sample results
                print(f"\n--- Sample {idx+1} (Person ID: {row['Person_id']}) ---")
                print(f"Generated: {generated_text}")
                print(f"Reference: {reference_text}")
                print("Metrics:")
                for metric_name, value in metrics.items():
                    print(f"  {metric_name}: {value:.4f}")
                print("-" * 50)
        
        # Clear GPU memory periodically
        if device == torch.device("cuda") and idx % 20 == 0:
            torch.cuda.empty_cache()
    
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
        results_df['Person_ID'] = person_ids
        results_df['Generated_Text'] = generated_texts
        results_df['Reference_Text'] = reference_texts
        results_df.to_csv('evaluation_results.csv', index=False)
        print("\nDetailed results saved to 'evaluation_results.csv'")
    else:
        print("Evaluation failed: No metrics were calculated.")

# Helper functions for ROUGE calculation
def calculate_rouge_n(ref_tokens, gen_tokens, n=1):
    """Calculates ROUGE-N F1 score based on n-gram overlap."""
    if len(ref_tokens) == 0 or len(gen_tokens) == 0:
        return 0.0
    
    def get_ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    
    ref_ngrams = get_ngrams(ref_tokens, n)
    gen_ngrams = get_ngrams(gen_tokens, n)
    
    if not ref_ngrams or not gen_ngrams:
        return 0.0
    
    ref_ngram_counts = {}
    for ngram in ref_ngrams:
        ref_ngram_counts[ngram] = ref_ngram_counts.get(ngram, 0) + 1
    
    gen_ngram_counts = {}
    for ngram in gen_ngrams:
        gen_ngram_counts[ngram] = gen_ngram_counts.get(ngram, 0) + 1
    
    overlap_count = 0
    for ngram, count in gen_ngram_counts.items():
        overlap_count += min(count, ref_ngram_counts.get(ngram, 0))
    
    precision = overlap_count / len(gen_ngrams) if gen_ngrams else 0.0
    recall = overlap_count / len(ref_ngrams) if ref_ngrams else 0.0
    
    if precision + recall == 0:
        return 0.0
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def calculate_rouge_l(ref_tokens, gen_tokens):
    """Calculates ROUGE-L F1 score based on LCS."""
    m, n = len(ref_tokens), len(gen_tokens)
    if m == 0 or n == 0: return 0.0
    
    L = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i-1] == gen_tokens[j-1]:
                L[i][j] = L[i-1][j-1] + 1
            else:
                L[i][j] = max(L[i-1][j], L[i][j-1])
    lcs_len = L[m][n]
    
    recall = lcs_len / m if m > 0 else 0.0
    precision = lcs_len / n if n > 0 else 0.0
    
    if recall + precision == 0:
        f1 = 0.0
    else:
        f1 = (2 * recall * precision) / (recall + precision)
    return f1

if __name__ == "__main__":
    evaluate_test_set()
