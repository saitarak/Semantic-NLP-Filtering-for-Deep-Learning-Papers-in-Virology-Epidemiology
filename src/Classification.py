import pandas as pd
from transformers import AutoTokenizer, AutoModel
from transformers import pipeline

filtered_data = pd.read_csv("../data/relevant_data.csv")
print(filtered_data)

# Initialize a zero-shot classification pipeline with a suitable model for scientific text classification
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define labels
labels = ["computer vision", "text mining", "both", "other"]

# Define function to classify text and return the label with the highest score
def classify_text(text):
    result = classifier(text, labels)
    # Get the label with the highest score
    highest_score_label = result['labels'][0]  # The first label is the highest-scoring one
    return highest_score_label

# Apply the classification function to each row's 'cleaned_text' and store the result in a new 'label' column
filtered_data['label'] = filtered_data['cleaned_text'].apply(classify_text)
print('filtered_data with classification:', filtered_data)    # The dataframe with data after labeling

filtered_data.to_csv("../data/classication_data.csv")    # Saving Dataframe after classification