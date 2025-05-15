import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

df = pd.read_csv("IMDB Dataset.csv", encoding='ISO-8859-1', usecols=['review', 'sentiment'])

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', ' ', text)                   # 去除 HTML 标签
    text = re.sub(r'[^\w\s]', '', text)                  # 去除标点符号
    text = re.sub(r'\d+', '', text)                      # 去除数字
    stop_words = set(stopwords.words('english'))         # 英文停用词
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

df['cleaned_review'] = df['review'].apply(clean_text)

df.to_csv("IMDB_cleaned.csv", index=False)
