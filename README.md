# Semantic-NLP-Filtering-for-Deep-Learning-Papers-in-Virology-Epidemiology
Overview
This project aims to:

Filter scientific papers to identify those relevant to deep learning approaches in virology and epidemiology.
Classify relevant papers into specific categories like "computer vision," "text mining," or both.
Extract and Report the specific deep learning method used in each relevant paper, such as "convolutional neural network" or "transformer."
The project employs various NLP techniques and transformer-based embeddings to achieve these tasks with high accuracy.

Data Processing Pipeline
The pipeline involves the following main stages:

Data Preprocessing: Text is cleaned by converting to lowercase, removing numbers, punctuation, non-word characters, and stop words. This step reduces noise in the text and improves embedding accuracy.

Filtering: Only papers that meet specific criteria are retained for analysis. This step involves computing semantic similarity between the abstract and predefined deep learning keywords related to virology and epidemiology.

Classification: Each relevant paper is classified based on the topic it covers, such as "computer vision" or "text mining."

Method Extraction: Specific deep learning techniques or methods (e.g., "CNN," "transformer") are extracted from each abstract.

Filtering Approach
For filtering, we use transformer-based embeddings to compute semantic similarity between the abstracts and a list of predefined keywords related to deep learning and virology. Specifically:

Embeddings Generation: We use Bio_ClinicalBERT to generate embeddings for both the paper abstracts and the keyword list.
Similarity Check: The similarity between each abstract’s embedding and the mean of keyword embeddings is computed using cosine similarity.
Thresholding: Only papers with a similarity score above a set threshold (0.5 in this case) are considered relevant. This semantic filtering approach captures more context than traditional keyword-based filtering by using contextual embeddings.
Why This Approach is More Effective than Keyword-Based Filtering
Unlike simple keyword-based filtering, which can miss relevant papers if specific terms aren’t present, this semantic similarity approach:

Captures Context: Embeddings take into account the entire context of each paper and keywords, not just exact matches.
Increases Recall: Papers that are semantically similar to the keywords but don’t contain exact matches are still captured.
Reduces Noise: Papers unrelated to deep learning in virology are more accurately filtered out.
Classification Approach
The filtered papers are then classified using a zero-shot classification model (facebook/bart-large-mnli). Labels such as "computer vision," "text mining," "both," and "other" are used to categorize each paper. Zero-shot classification is chosen to allow flexible, label-free categorization.

Method Extraction Approach
To extract specific methods used in each paper:

Rule-Based Pattern Matching: We define a dictionary of regex patterns for deep learning techniques, including "convolutional neural network," "transformer," and "text mining."
Matching Specific Methods: Each abstract is scanned against these patterns, and the matching method is reported. If no match is found, a general label like "Deep learning based" is assigned.
This approach ensures that each paper is associated with a precise technique, fulfilling the task’s requirement for specific method extraction.

Results and Dataset Statistics
The system outputs three main datasets:

Filtered Dataset: Contains only papers relevant to deep learning in virology/epidemiology.
Classified Dataset: Each paper is categorized as "computer vision," "text mining," "both," or "other."
Method Extraction Dataset: The specific method used in each paper is extracted, providing insight into the techniques discussed.
Example Statistics
Total papers in original dataset: X
Papers retained after filtering: Y
Papers categorized by class:
Computer Vision: A
Text Mining: B
Both: C
Other: D
Methods Extracted:
Convolutional Neural Network: X1
Transformer: X2
... (list all other methods)
Installation and Usage
Prerequisites
Ensure that you have the following installed:

Python 3.6 or higher
Required packages in requirements.txt
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/your_username/your_repository.git
cd your_repository
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Running the Pipeline
Data Preprocessing, Filtering, and Classification

bash
Copy code
python src/data_processing.py
Method Extraction

bash
Copy code
python src/method_extraction.py
Results will be saved to the data/ directory.

Contributing
Contributions are welcome! Please submit a pull request with clear descriptions and examples.

License
This project is licensed under the MIT License. See the LICENSE file for details.
