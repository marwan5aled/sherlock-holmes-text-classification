import pandas as pd
import numpy as np
import re
import os
import requests
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier

# تنزيل الكتاب إذا لم يكن موجودًا
if not os.path.exists('sherlock-holmes.txt'):
    text = requests.get('https://www.gutenberg.org/files/1661/1661-0.txt').text
    with open("sherlock-holmes.txt", "w", encoding='utf-8') as text_file:
        text_file.write(text)

# قراءة الكتاب
text = open('sherlock-holmes.txt', 'r', encoding='utf-8').read()

# معالجة النص وتقسيمه إلى جمل
stop_pattern = r'\.|\?|\!'
sentences = re.split(stop_pattern, text)
sentences = [re.sub("\r|\n", " ", s.lower()) for s in sentences][3:]

# التأكد من وجود ملف التسمية
csv_file = 'D:/vscode/sherlock-holmes-annotations.csv'  # استخدم المسار المطلق للملف
if not os.path.exists(csv_file):
    raise FileNotFoundError(f"File {csv_file} does not exist.")

# تحميل ملف التسمية
df = pd.read_csv(csv_file)

# التأكد من تطابق عدد الجمل مع عدد الصفوف في CSV
num_sentences = len(sentences)
num_rows = len(df)

if num_sentences != num_rows:
    if num_sentences > num_rows:
        sentences = sentences[:num_rows]
    else:
        sentences.extend([''] * (num_rows - num_sentences))

df['text'] = sentences

# تحضير التسميات وإنشاء إطار بيانات
labels = np.zeros(df.shape[0])
labels[(df['has_sherlock'] == True)] = 1
labels[(df['has_watson'] == True)] = 2
df['labels'] = labels
df = df[df['labels'] != 0]
X = df['text'].values
y = df['labels'].values

# تقسيم البيانات إلى مجموعة تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# تدريب نموذج تصنيف النص
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='hinge', penalty='l2',
                          alpha=1e-3, random_state=42,
                          max_iter=5, tol=None))
])

# تدريب النموذج
text_clf.fit(X_train, y_train)

# تقييم أداء النموذج
predicted = text_clf.predict(X_test)
print(np.mean(predicted == y_test))

print(metrics.classification_report(y_test, predicted,
      target_names=['sherlock', 'watson']))
