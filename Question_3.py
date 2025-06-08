# Q3: Text classification using 20 Newsgroups dataset and knowledge taxonomy
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Define category mapping
category_map = {
    'misc.forsale': 'Sales',
    'rec.motorcycles': 'Motorcycles',
    'rec.sport.baseball': 'Baseball',
    'sci.crypt': 'Cryptography',
    'sci.space': 'Space'
}
categories = list(category_map.keys())

# Load the training subset of the dataset
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)

# Vectorize the training text using TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(newsgroups_train.data)

# Train a Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train_tfidf, newsgroups_train.target)

# Define input sentences for prediction
input_data = [
    "The curveballs of right handed pitchers tend to curve to the left",
    "Caesar cipher is an ancient form of encryption",
    "This two-wheeler is really good on slippery roads"
]

# Vectorize and predict the category of input sentences
X_input = vectorizer.transform(input_data)
predicted = clf.predict(X_input)

# Print the input sentence with its predicted category
for sentence, index in zip(input_data, predicted):
    category = newsgroups_train.target_names[index]
    readable_category = category_map[category]
    print(f"Input: {sentence}\nPredicted Category: {readable_category}\n")
