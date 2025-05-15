import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def polarity_to_score(p):
    if p >= 0.6:
        return 9.5
    elif p >= 0.3:
        return 8
    elif p >= 0.1:
        return 6
    elif p > -0.1:
        return 5
    elif p > -0.3:
        return 4
    elif p > -0.6:
        return 2.5
    else:
        return 1

def sketchengine_score(text):
    words = text.lower().split()
    scores = [keyword_score_dict[word] for word in words if word in keyword_score_dict]
    return round(sum(scores) / len(scores), 2) if scores else 5.0


df = pd.read_csv("IMDB_cleaned.csv")
X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned_review'], df['sentiment'], test_size=0.2, random_state=1
)
model = make_pipeline(TfidfVectorizer(max_features=5000), LinearSVC())
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

vectorizer = TfidfVectorizer(max_features=20)
X_tfidf = vectorizer.fit_transform(df['cleaned_review'])
feature_names = vectorizer.get_feature_names_out()
print("\U0001f50d Top 20 TF-IDF 特征词：")
print(feature_names)

sketch_df = pd.read_csv("poswordlist_user_sc22yc3_imdb_20250514043146.csv")  # 包含 'Item', 'Frequency'
sketch_df['polarity'] = sketch_df['Item'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
sketch_df['auto_score'] = sketch_df['polarity'].apply(polarity_to_score)

keyword_score_dict = dict(zip(sketch_df['Item'], sketch_df['auto_score']))

df['sk_score_auto'] = df['cleaned_review'].apply(sketchengine_score)
df.to_csv("IMDB_with_sketch_autoscore.csv", index=False)
print("已基pr基于 TextBlob 极性的情绪打分，结果已保存。")

df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
df['sk_pred'] = df['sk_score_auto'].apply(lambda x: 1 if x > 5 else 0)

acc = accuracy_score(df['label'], df['sk_pred'])
prec = precision_score(df['label'], df['sk_pred'])
rec = recall_score(df['label'], df['sk_pred'])
f1 = f1_score(df['label'], df['sk_pred'])

# Diagram 1
plt.figure(figsize=(8, 5))
sns.histplot(df['sk_score_auto'], bins=20, kde=True, color='dodgerblue')
plt.title("Auto Score Distribution")
plt.xlabel("Auto Score (0–10)")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()

# Diagram 2
plt.figure(figsize=(7, 5))
sns.boxplot(x='sentiment', y='sk_score_auto', data=df, palette='Set2')
plt.title("Score Distribution by Sentiment")
plt.xlabel("True Sentiment")
plt.ylabel("SketchEngine Auto Score")
plt.grid(True)
plt.tight_layout()
plt.show()

print(" SketchEngine Auto Score Evaluation Results")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1 Score : {f1:.4f}")

cm = confusion_matrix(y_test, y_pred, labels=["positive", "negative"])

# Confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Positive", "Negative"],
            yticklabels=["Positive", "Negative"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()
