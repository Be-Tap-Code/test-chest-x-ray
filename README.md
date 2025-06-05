# Chest X-ray Report Generation

This project implements a deep learning model to generate medical reports from chest X-ray images. It leverages CLIP for image feature extraction and BioBART for report generation, connected by an MLP. The application provides a Streamlit interface for interactive testing and an evaluation script for comprehensive metric calculation.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Running the Streamlit App on Kaggle](#running-the-streamlit-app-on-kaggle)
- [Running the Evaluation Script](#running-the-evaluation-script)
- [Model Checkpoint and Data](#model-checkpoint-and-data)
- [Metrics](#metrics)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Image Feature Extraction:** Utilizes CLIP (Contrastive Language-Image Pre-training) to extract meaningful features from chest X-ray images.
- **Report Generation:** Employs BioBART (a BART model fine-tuned on biomedical text) to generate descriptive medical reports based on image features.
- **Multilayer Perceptron (MLP):** Connects the image features from CLIP to the input space of BioBART.
- **Interactive Streamlit App:** A user-friendly web interface for selecting and testing individual image pairs from the test dataset and viewing generated reports alongside reference reports.
- **Comprehensive Evaluation Script:** A standalone script to evaluate the model's performance on the entire test set using various metrics (BLEU, METEOR, ROUGE).
- **Kaggle Compatibility:** Optimized for execution on Kaggle notebooks, leveraging GPU resources.

## Project Structure

```
.
├── app.py                     # Streamlit application for interactive testing
├── evaluate.py                # Script for comprehensive model evaluation
├── requirements.txt           # Python dependencies
└── README.md                  # Project README file
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Be-Tap-Code/test-chest-x-ray.git
    cd test-chest-x-ray
    ```

2.  **Install dependencies:**
    It's recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

## Running the Streamlit App on Kaggle

To run the interactive Streamlit application on Kaggle, follow these steps:

1.  **Create a new Kaggle Notebook.**
2.  **Add necessary data:**
    - Go to the "Data" tab (usually on the right sidebar).
    - Click "Add Data" and search for the following Kaggle datasets:
        - `best-model-iux-ray` (contains `vit_biobart_best_model.pt` and `xray_images` folder)
        - `data-split-csv` (contains `Test_Data.csv`)
    - Ensure these datasets are mounted to `/kaggle/input/`.

3.  **Run the following commands in a Kaggle Notebook cell:**

    ```python
    # Clone repository
    !git clone https://github.com/Be-Tap-Code/test-chest-x-ray.git
    %cd test-chest-x-ray

    # Install requirements
    !pip install -r requirements.txt

    # Run Streamlit with localtunnel for external access
    !streamlit run app.py &>/logs.txt & npx localtunnel --port 8501

    # Get the URL (may take a moment to appear)
    !cat /logs.txt
    ```

    You should see an output similar to this: `your url is: https://[random-string].loca.lt`. Copy this URL and open it in your browser to interact with the Streamlit app.

## Running the Evaluation Script

To run the comprehensive evaluation script on Kaggle:

1.  **Create a new Kaggle Notebook or use an existing one.**
2.  **Add necessary data:**
    - Go to the "Data" tab.
    - Click "Add Data" and search for the following Kaggle datasets:
        - `best-model-iux-ray` (contains `vit_biobart_best_model.pt` and `xray_images` folder)
        - `data-split-csv` (contains `Test_Data.csv`)
    - Ensure these datasets are mounted to `/kaggle/input/`.

3.  **Run the following commands in a Kaggle Notebook cell:**

    ```python
    # Clone repository
    !git clone https://github.com/Be-Tap-Code/test-chest-x-ray.git
    %cd test-chest-x-ray

    # Install requirements
    !pip install -r requirements.txt

    # Run the evaluation script
    !python evaluate.py
    ```

    The script will print the evaluation metrics for each sample and the final aggregate metrics to the console. A file named `evaluation_results.csv` will also be generated in your Kaggle working directory, containing detailed results for each processed sample.

## Model Checkpoint and Data

The models are pre-trained and the checkpoint (`vit_biobart_best_model.pt`) along with the X-ray images (`xray_images` folder) are expected to be available in the `/kaggle/input/best-model-iux-ray/` directory on Kaggle. The test data split (`Test_Data.csv`) is expected in `/kaggle/input/data-split-csv/`.

- **Model Checkpoint:** `vit_biobart_best_model.pt` (part of `best-model-iux-ray` Kaggle dataset)
- **X-ray Images:** `/kaggle/input/best-model-iux-ray/xray_images/`
- **Test Data CSV:** `/kaggle/input/data-split-csv/Test_Data.csv`

## Metrics

The evaluation script calculates the following metrics for each sample and provides their mean and standard deviation across the entire test set:

-   **BLEU (Bilingual Evaluation Understudy):** BLEU-1, BLEU-2, BLEU-3, BLEU-4
-   **METEOR (Metric for Evaluation of Translation with Explicit Ordering):** Based on a harmonic mean of precision and recall.
-   **ROUGE (Recall-Oriented Understudy for Gisting Evaluation):** ROUGE-1 (unigram overlap), ROUGE-2 (bigram overlap), ROUGE-3 (trigram overlap), ROUGE-4 (4-gram overlap), ROUGE-L (Longest Common Subsequence).

## Contributing

Feel free to open issues or submit pull requests if you have suggestions, bug fixes, or new features.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details (if you have one, otherwise remove this section). 