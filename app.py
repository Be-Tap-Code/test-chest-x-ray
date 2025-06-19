import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings

import streamlit as st

# Configure Streamlit page
st.set_page_config(
    page_title="Chest X-ray Report Generation",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .stButton>button {
        width: 100%;
    }
    .stSelectbox {
        width: 100%;
    }
    .stMarkdown {
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

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

# Download required NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

# Configure logging to only show errors
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress specific CUDA warnings
if torch.cuda.is_available():
    torch.cuda.set_device(0)  # Use first GPU
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    # Suppress CUDA warnings
    torch.cuda.empty_cache()
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

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

@st.cache_resource
def load_models():
    # Load CLIP model and processor with from_tf=True
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", from_tf=True).to(device).eval()
    feature_extractor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Load BioBART tokenizer and model
    biobart_model_name = "GanjinZero/biobart-base"
    biobart_tokenizer = BartTokenizer.from_pretrained(biobart_model_name)
    biobart_model = BartForConditionalGeneration.from_pretrained(biobart_model_name).to(device)

    # Initialize MLP with correct input dimension
    mlp_input_dim = 1536  # Updated to match the checkpoint's dimension
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
        st.error(f"Checkpoint not found at {checkpoint_path}")
        return None, None, None, None, None

    return clip_model, feature_extractor, biobart_model, biobart_tokenizer, mlp

@st.cache_data
def load_test_data():
    # Load test dataset
    test_dataset = pd.read_csv('/kaggle/input/data-split-csv/Test_Data.csv')
    return test_dataset

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

def main():
    st.title("Chest X-ray Report Generation")
    st.write("Select a test sample to generate a medical report.")

    # Load models
    with st.spinner("Loading models..."):
        clip_model, feature_extractor, biobart_model, biobart_tokenizer, mlp = load_models()
        if None in [clip_model, feature_extractor, biobart_model, biobart_tokenizer, mlp]:
            st.error("Failed to load models. Please check the model checkpoint.")
            return

    # Load test data
    test_dataset = load_test_data()
    
    # Create a list of sample options
    sample_options = [f"Sample {i+1} (Person ID: {row['Person_id']})" 
                     for i, row in test_dataset.iterrows()]
    
    # Add sample selector
    selected_sample = st.selectbox("Select a test sample:", sample_options)
    
    if selected_sample:
        # Get the index of selected sample
        sample_idx = sample_options.index(selected_sample)
        selected_row = test_dataset.iloc[sample_idx]
        
        # Load images
        image1 = load_image(selected_row['Image1'])
        image2 = load_image(selected_row['Image2'])
        
        if image1 is not None and image2 is not None:
            # Display images
            col1, col2 = st.columns(2)
            with col1:
                st.image(image1, caption="First X-ray Image", use_column_width=True)
            with col2:
                st.image(image2, caption="Second X-ray Image", use_column_width=True)
            
            if st.button("Generate Report"):
                with st.spinner("Generating report..."):
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
                        report = generate_report(feature_tensor, mlp, biobart_model, biobart_tokenizer)
                        
                        # Display reports
                        st.subheader("Generated Report")
                        st.write(report)
                        
                        st.subheader("Reference Report")
                        st.write(selected_row['Report'])
                    else:
                        st.error("Failed to extract features from images.")
        else:
            st.error("Failed to load one or both images.")

if __name__ == "__main__":
    main()
