import matplotlib.pyplot as plt

# Data for papers categorized by class
categories = ["Computer Vision", "Text Mining", "Both", "Other"]
category_counts = [106, 405, 2053, 2742]

# Data for methods extracted
methods = [
    "Deep learning relevant(but not defined)", "Deep Learning", "Language Modeling",
    "Convolutional Neural Network", "Text Mining", "Recurrent Neural Network",
    "Image Processing", "Feedforward Neural Network", "Vision Transformer",
    "Object Recognition", "Generative AI", "Large Language Model", "GPT (Transformer based)",
    "BERT (Transformer based)", "Computational Linguistics", "Transformer Architecture"
]
method_counts = [2677, 1037, 442, 322, 238, 187, 116, 86, 76, 40, 33, 20, 19, 8, 4, 1]

# Plotting: Papers categorized by class (Pie Chart)
plt.figure(figsize=(8, 8))
plt.pie(category_counts, labels=categories, autopct='%1.1f%%', startangle=140)
plt.title("Classification of relevant papers")
plt.show()

# Plotting: Methods Extracted (Bar Chart)
plt.figure(figsize=(12, 6))
plt.barh(methods, method_counts, color='skyblue')
plt.xlabel("Number of Papers")
plt.ylabel("Method")
plt.title("Distribution of Methods Extracted from Papers")
plt.gca().invert_yaxis()
plt.show()