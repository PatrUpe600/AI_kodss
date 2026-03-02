import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import gradio as gr

data = pd.read_csv("Text_Emotion.csv")

#convert emojies into binary classes
data['emotion'] = data['emotion'].map({
    '☹️' : 0,
    '🙂' : 1
})

data.head()

data.isnull().sum()

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.lower().strip()

data["text"] = data["text"].astype(str).apply(clean_text)

#split the data to train and test 
X_train, X_test, y_train, y_test = train_test_split(
    data["text"], data["emotion"], test_size=0.2, random_state=42
)

#vectorization
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))

def analyze_and_reply(text):
    text_clean = clean_text(text)
    X_input = vectorizer.transform([text_clean])
    prediction = model.predict(X_input)[0]
    
    if prediction == 1:
        return f"Prediction: 😊 Happy\nResponse: I'm glad to hear that! thanx for your review!"
    else:
        return f"Prediction: 😔 Sad\nResponse: I'm sorry to hear that. your feedback will be considered"
    
iface = gr.Interface(
    fn=analyze_and_reply,
    inputs="text",
    outputs="text",
    title="Emotion Detector (Happy or Sad)",
    description="Enter a short review or sentence, and the model will detect if it's happy or sad, then reply to you.",
)

iface.launch()    