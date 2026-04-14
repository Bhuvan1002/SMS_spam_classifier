import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split 
from  sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

#to open and readd the csv file
df = pd.read_csv('spam.csv',encoding = 'latin-1')

#to use only the columns reequired
df = df[['v1','v2']]
df.columns = ['labels','message']
print (df.head())

#to count the ham as 0 and spam as 1 for machine to ubderstand
df['labels']=df['labels'].str.strip().str.lower()
df['label'] = df['labels'].map({'ham':0,'spam':1})
df = df.dropna(subset=['label'])  # Drop any rows with NaN labels

print(df['label'].unique())


#to clean all the special char and use only the text required
def clean_text(text):
    text=text.lower()
    text=re.sub(r'[^a-zA-Z0-9]',' ',text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text 

df = df .dropna(subset =['label'])

#convertinf text into number 
df['message']=df['message'].apply(clean_text)
vectorizer = TfidfVectorizer(ngram_range=(1,2),stop_words='english')
x = vectorizer.fit_transform(df['message'])
y = df['label']

print(df['label'].isnull().sum())


# to train and test the data
# random to test every time  
# test_size to train 80 and test 20 %
x_train,x_test,y_train,y_test = train_test_split (
    x,y,test_size = 0.2,random_state = 42
    )

#create a model
#train model by looking at it input and output
model= MultinomialNB()
model.fit(x_train,y_train)


#fucntion to predict the message
#with the same text cleaning and vectorization 
def predict_spam(text):
    text=clean_text(text)
    text=vectorizer.transform([text])
    prediction = model.predict(text)
    return "spam" if prediction[0] == 1 else "not spam"

#predict using the model and stores it
#compares it with the real answer 
#checks the accuracy of the model
y_pred = model.predict(x_test)
accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy_score(y_test,y_pred))

msg = input ("Enter a message:")
result = predict_spam(msg)
print("result:",result)
