import pandas as pd
import re

filtered_data = pd.read_csv('../data/method_extraction.csv')

method_patterns = {
    "Convolutional Neural Network": r"\b(convolutional neural network|CNN)\b",
    "Recurrent Neural Network": r"\b(recurrent neural network|RNN|long short-term memory network|LSTM|GRNN)\b",
    "Feedforward Neural Network": r"\b(feedforward neural network|neural net algorithm|multilayer perceptron)\b",
    "Deep Learning": r"\b(deep learning|deep neural networks)\b",
    "Object Recognition": r"\b(object recognition|image recognition|scene understanding)\b",
    "Image Processing": r"\b(image processing|vision algorithms|computer graphics and vision)\b",
    "Text Mining": r"\b(text mining|textual data analysis|text data analysis|text analytics|text analysis)\b",
    "Language Modeling": r"\b(language modeling|language processing|computational semantics)\b",
    "Computational Linguistics": r"\b(computational linguistics|speech and language technology)\b",
    "Transformer Architecture": r"\b(transformer architecture|self-attention models|attention-based neural networks|sequence-to-sequence models)\b",
    "BERT": r"\b(BERT|transformer models)\b",
    "GPT": r"\b(GPT|generative pretrained transformer)\b",
    "Generative AI": r"\b(generative artificial intelligence|generative AI|generative deep learning|generative models|GAN|VAE)\b",
    "Large Language Model": r"\b(large language model|llm|transformer-based model|pretrained language model|"
                            r"generative language model|foundation model|state-of-the-art language model)\b",
    "Vision Transformer": r"\b(vision transformer|multimodal model|multimodal neural network|diffusion model|generative diffusion model|"
                          r"diffusion based generative model|continuous diffusion model)\b"
}

# Function to detect method used in the abstract
def extract_method_rule_based(text):
    for method, pattern in method_patterns.items():
        if re.search(pattern, text, re.IGNORECASE):
            return method
    return "Deep learning based"  # Return 'deep learning based' if no pattern matches

# Apply the rule-based function to each abstract
filtered_data['extracted_method'] = filtered_data['cleaned_text'].apply(extract_method_rule_based)

print("filtered_data with method extraction:",filtered_data)    # The dataframe with data after method extraction

filtered_data.to_csv("../data/method_extraction.csv")    # Saving dataframe after method extraction
