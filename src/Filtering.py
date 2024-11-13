import pandas as pd
import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch

# Data Preprocessing steps
df = pd.read_csv('../data/collection_with_abstracts.csv')
df['text'] = df['Title'].fillna('') + " " + df['Abstract'].fillna('')
df = df[['text']].dropna()

def preprocess_text(text):
    text = text.lower()    # Make all Lowercase letters
    text = re.sub(r'\d+', '', text)    # Removal of numbers
    text = re.sub(r'[^\w\s]', ' ', text)    # Removal of punctuation and non-word characters
    text = re.sub(r'\s+', ' ', text).strip()    # Raplace multiple spaces with single space
    words = word_tokenize(text)    # Text tokenization
    words = [word for word in words if word not in stopwords.words('english')]    # Stopword removal
    return ' '.join(words)

df['cleaned_text'] = df['text'].apply(preprocess_text)

#Filter papers that not meet the criteria of utilizing deep learning approaches in virology/epidemiology
keywords = [
    "neural network", "artificial neural network", "machine learning model", 
    "feedforward neural network", "neural net algorithm", "multilayer perceptron",
    "convolutional neural network", "recurrent neural network", "long short-term memory network",
    "CNN", "GRNN", "RNN", "LSTM", "deep learning", "deep neural networks", "artificial intelligence",
    "virology", "epidemiology", "supervised learning", "unsupervised learning", "gradient descent",
    "predictive modeling", "classification", "feature extraction", "disease prediction", "computational biology",
    "drug discovery", "clinical", "health informatics", "bioinformatics", "clustering",
    "viral diseases", "vaccine", "virus evolution", "host virus", "pathogenesis", "molecular virology",
    "pathology", "viral replication", "immune response", "phylogenetics", "preclinical studies", "viral lifecycle", 
    "viral particle morphology", "zonotic diseases", "viral infections", "immunization", "vaccine efficacy", 
    "Analytic epidemiology", "Applied epidemiology", "epidemic", "prevalance", "morbidity", "epidemic curve", 
    "virulence", "validity", "incidence", "epidemic", "endemic", "morality", "morbidity", "infection rate",
    "health monitoring", "vaccination rate", "health monitoring"
]

tokenizer = AutoTokenizer.from_pretrained("tarasophia/Bio_ClinicalBERT_medical")
model = AutoModel.from_pretrained("tarasophia/Bio_ClinicalBERT_medical")

def get_embedding(text):
    # Tokenize the input text and convert to PyTorch tensors
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    # Pass the inputs through the model to get the outputs
    with torch.no_grad():  # Disable gradient calculation to save memory
        outputs = model(**inputs)
    
    # Take the mean of the last hidden state to get a single vector embedding
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    
    return embedding

keyword_embeddings = []
for keyword in keywords:
    keyword_embedding = get_embedding([keyword])
    keyword_embeddings.append(keyword_embedding)
mean_keyword_embedding = np.mean(keyword_embeddings, axis=0)    # Mean of keyword embeddings

indices = []
abstract_embeddings = []
for index, abstract in enumerate(df['cleaned_text']):
    abstract_embedding = get_embedding([abstract])
    abstract_embedding = np.reshape(abstract_embedding, (1, -1))
    mean_keyword_embedding = np.reshape(mean_keyword_embedding, (1, -1))
    similarity = cosine_similarity(abstract_embedding, mean_keyword_embedding)[0][0]    # Semantic similarity check between mean keyword embeddings and abstract embeddings
    if similarity > 0.5:
        print(f"Abstract {index+1} is relevant with similarity: {similarity}")
        indices.append(index)
print("number of indices:", len(indices))    # Number of papers after filtering
print('Dataframe with relevant papers:', df)    # Dataframe with data after filtering

df['cleaned_text'][indices[:]].to_csv("../data/relevant_data.csv")    # Saving Dataframe after filtering papers that do not meet the criteria
