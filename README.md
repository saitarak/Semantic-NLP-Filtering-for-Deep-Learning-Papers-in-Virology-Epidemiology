# Semantic-NLP-Filtering-for-Deep-Learning-Papers-in-Virology-Epidemiology

## Overview
This project aims to:

Filter scientific papers to identify those relevant to deep learning approaches in virology and epidemiology.
Classify relevant papers into specific categories like "computer vision," "text mining."
Extract and Report the name of the method used for each relevant paper.
The project employs various NLP techniques and transformer-based embeddings to achieve these tasks with high accuracy.

## Data Processing Pipeline
The pipeline involves the following main stages:

### Data Preprocessing 
Text is cleaned by converting to lowercase, removing numbers, punctuation, non-word characters, and stop words. This step reduces noise in the text and improves embedding accuracy.

### Filtering Approach
For filtering, we use transformer-based embeddings to compute semantic similarity between the abstracts and a mean of predefined keywords related to deep learning approaches in virology.

Embeddings Generation: We use Bio_ClinicalBERT to generate embeddings for both the paper abstracts and the keyword list and compared using semantic similarity search.
Thresholding: Only papers with a similarity score above a set threshold are considered relevant. This semantic filtering approach captures more context than traditional keyword-based filtering by using contextual embeddings.
Why This Approach is More Effective than Keyword-Based Filtering
Unlike simple keyword-based filtering, which can miss relevant papers if specific terms aren’t present, this semantic similarity approach:

Captures Context: Embeddings take into account the entire context of each paper and keywords, not just exact matches.
Increases Recall: Papers that are semantically similar to the keywords but don’t contain exact matches are still captured.
Reduces Noise: Papers unrelated to deep learning in virology are more accurately filtered out.

### Classification Approach
The filtered papers are then classified using a zero-shot classification model (facebook/bart-large-mnli). Labels such as "computer vision," "text mining," "both," and "other" are used to categorize each paper. Zero-shot classification using scientific language model is chosen to allow flexible, label-free categorization.

### Method Extraction Approach
To extract specific methods used in each paper:

Rule-Based Pattern Matching: We define a dictionary of regex patterns, including "convolutional neural network," "transformer," and "text mining" and more.
Matching Specific Methods: Each abstract is scanned against these patterns, and the matching method is reported. .
This approach ensures that each paper is associated with a precise technique, fulfilling the requirement of specific method extraction.

## Dataset Statistics
The system statistics are as follows: <br />
Total papers in original dataset: 11450 papers <br />
Papers retained after filtering: 5306 papers <br />

Papers categorized by class: <br />
Computer Vision: 106 <br />
Text Mining: 405 <br />
Both: 2053 <br />
Other: 2742 <br />

Methods Extracted: <br />
Deep learning relevant(but not defined)      2677 <br />
Deep Learning                                1037 <br />
Language Modeling                            442 <br />
Convolutional Neural Network                 322 <br />
Text Mining                                  238 <br />
Recurrent Neural Network                     187 <br />
Image Processing                             116 <br />
Feedforward Neural Network                   86 <br />
Vision Transformer                           76 <br />
Object Recognition                           40 <br />
Generative AI                                33 <br />
Large Language Model                         20 <br />
GPT (Transformer based)                      19 <br />
BERT (Transformer based)                     8 <br />
Computational Linguistics                    4 <br />
Transformer Architecture                     1 <br />

For more statistics you can refer the data folder to have a look at method_extraction.csv

## Running the Pipeline
Filtering
```
python Filtering.py
```

Classification
```
python classification.py
```

Method Extraction
```
python Method_extraction.py
```
csv files will be saved to the data directory.

