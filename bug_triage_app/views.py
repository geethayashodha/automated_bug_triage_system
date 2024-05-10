from django.shortcuts import render,redirect
from .forms import UserCreform,BugFeatureForm
from .models import User,BugFeature
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')

model = joblib.load('model.joblib')
vectorizer = joblib.load('vectorizer.joblib')

def preprocess_text(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word.lower() for word in tokens if word.isalnum() and word.lower() not in stop_words]
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens] 
    return stemmed_tokens, lemmatized_tokens

def predict_text(product, component, summary, model, vectorizer):
    text = ' '.join([product, component, summary])
    stemmed_tokens, lemmatized_tokens = preprocess_text(text)
    preprocessed_text = ' '.join(stemmed_tokens)  
    features = vectorizer.transform([preprocessed_text])
    prediction = model.predict(features)
    return prediction

def home(request):
    return render(request,'html/index.html')

def register(request):
    if request.method == "POST":
        f = UserCreform(request.POST)
        if f.is_valid():
            f.save()
            return redirect('lg')
    else:
        f = UserCreform()
    return render(request,'html/register.html',{'g':f})

def report_features(request):
    if request.method == 'POST':
        form = BugFeatureForm(request.POST)
        if form.is_valid():
            product = form.cleaned_data['product']
            component = form.cleaned_data['component']
            summary = form.cleaned_data['summary']
            prediction = predict_text(product, component, summary, model, vectorizer)
            print(prediction)
            form.save()
            context = {'prediction': prediction}
            return render(request, 'html/result_feature.html', context)
    else:
        form = BugFeatureForm()
    return render(request, 'html/report_feature.html', {'form': form})

