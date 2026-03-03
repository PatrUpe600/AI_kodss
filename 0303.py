import pandas as pd
#  īstenojot tādas darbības kā datu kopu ielāde, izlīdzināšana, apvienošana un pārveidošana
import re
# nodrošina atbalstu darbam ar regulārajām izteiksmēm, ļaujot meklēt, saskaņot un manipulēt ar virknēm, izmantojot sarežģītus modeļu aprakstus

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
# atbalsta uzraudzītu un nepārraudzītu mācīšanos

import gradio as gr
# izveido tīmekļa lietojumprogrammu mašīnmācīšanās modeļa API

# Izlasa 
data = pd.read_csv("Text_Emotion.csv")

#convert emojies into binary classes / izveido emocijas "bēdīgs"-0, "laimīgs"-1
data['emotion'] = data['emotion'].map({
    '☹️' : 0,
    '🙂' : 1
})

# Izvada tabulu ar programmas precizitāti
data.head()
data.isnull().sum()

# Saraksta terkstu salasamā veidā
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.lower().strip()

data["text"] = data["text"].astype(str).apply(clean_text)

#split the data to train and test / Testē datus 
X_train, X_test, y_train, y_test = train_test_split(
    data["text"], data["emotion"], test_size=0.2, random_state=42
)

#vectorization / teksts->numuri
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Izmanto logīstisko regresiju modeli
model = LogisticRegression(max_iter=1000)

# Apgūst modeļus starp teksta elementiem.
model.fit(X_train_vec, y_train)

# Paredz testa datu etiķetes.
y_pred = model.predict(X_test_vec)

# Novērtē klasifikatora veiktspēju.
print(classification_report(y_test, y_pred))

# Analizē un izved 1 vai 0 / emociju
def analyze_and_reply(text):
    text_clean = clean_text(text)
    X_input = vectorizer.transform([text_clean])
    prediction = model.predict(X_input)[0]

# Apraksta emocijas    
    if prediction == 1:
        return f"Prediction: 😊 Happy\nResponse: YIPPEE!"
    else:
        return f"Prediction: 😔 Sad\nResponse: Oh naur."
    
# izveido web interfeisu    
iface = gr.Interface(
    fn=analyze_and_reply,
    inputs="text",
    outputs="text",
    title="Emotion Detector (Happy or Sad)",
    description="Enter a short review or sentence, and the model will detect if it's happy or sad, then reply to you.",
)

iface.launch()    