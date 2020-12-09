---
title: "NLP : LSTM, T-SNE, Word Embedding"
date: 2020-12-09
tags: [LSTM, Word Embedding]
header:
  
excerpt: "Deep Learning, Semantic Segmentation, Convolutional Neural Networks, SegNet, HDR Images, Tensorflow, Keras, Tensorboard"
mathjax: "true"
---

```
import csv
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

!wget --no-check-certificate \
    https://storage.googleapis.com/laurencemoroney-blog.appspot.com/bbc-text.csv \
    -O /tmp/bbc-text.csv
```



```
vocab_size = 1000
embedding_dim = 16
max_length = 120
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_portion = .8
```


```
sentences = []
labels = []
stopwords = [ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]
print(len(stopwords))
```

    153
    


```
with open("/tmp/bbc-text.csv", 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    for row in reader:
        labels.append(row[0])
        sentence = row[1]
        for word in stopwords:
            token = " " + word + " "
            sentence = sentence.replace(token, " ")
        sentences.append(sentence)

print(len(labels))
print(len(sentences))
print(sentences[0])
```

    2225
    2225
    tv future hands viewers home theatre systems  plasma high-definition tvs  digital video recorders moving living room  way people watch tv will radically different five years  time.  according expert panel gathered annual consumer electronics show las vegas discuss new technologies will impact one favourite pastimes. us leading trend  programmes content will delivered viewers via home networks  cable  satellite  telecoms companies  broadband service providers front rooms portable devices.  one talked-about technologies ces digital personal video recorders (dvr pvr). set-top boxes  like us s tivo uk s sky+ system  allow people record  store  play  pause forward wind tv programmes want.  essentially  technology allows much personalised tv. also built-in high-definition tv sets  big business japan us  slower take off europe lack high-definition programming. not can people forward wind adverts  can also forget abiding network channel schedules  putting together a-la-carte entertainment. us networks cable satellite companies worried means terms advertising revenues well  brand identity  viewer loyalty channels. although us leads technology moment  also concern raised europe  particularly growing uptake services like sky+.  happens today  will see nine months years  time uk   adam hume  bbc broadcast s futurologist told bbc news website. likes bbc  no issues lost advertising revenue yet. pressing issue moment commercial uk broadcasters  brand loyalty important everyone.  will talking content brands rather network brands   said tim hanlon  brand communications firm starcom mediavest.  reality broadband connections  anybody can producer content.  added:  challenge now hard promote programme much choice.   means  said stacey jolna  senior vice president tv guide tv group  way people find content want watch simplified tv viewers. means networks  us terms  channels take leaf google s book search engine future  instead scheduler help people find want watch. kind channel model might work younger ipod generation used taking control gadgets play them. might not suit everyone  panel recognised. older generations comfortable familiar schedules channel brands know getting. perhaps not want much choice put hands  mr hanlon suggested.  end  kids just diapers pushing buttons already - everything possible available   said mr hanlon.  ultimately  consumer will tell market want.   50 000 new gadgets technologies showcased ces  many enhancing tv-watching experience. high-definition tv sets everywhere many new models lcd (liquid crystal display) tvs launched dvr capability built  instead external boxes. one example launched show humax s 26-inch lcd tv 80-hour tivo dvr dvd recorder. one us s biggest satellite tv companies  directtv  even launched branded dvr show 100-hours recording capability  instant replay  search function. set can pause rewind tv 90 hours. microsoft chief bill gates announced pre-show keynote speech partnership tivo  called tivotogo  means people can play recorded programmes windows pcs mobile devices. reflect increasing trend freeing multimedia people can watch want  want.
    


```
train_size = int(len(sentences) * training_portion)

train_sentences = sentences[:train_size]
train_labels = labels[:train_size]

validation_sentences = sentences[train_size:]
validation_labels = labels[train_size:]

print(train_size)
print(len(train_sentences))
print(len(train_labels))
print(len(validation_sentences))
print(len(validation_labels))
```

    1780
    1780
    1780
    445
    445
    


```
tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_sentences)
word_index = tokenizer.word_index

train_sequences = tokenizer.texts_to_sequences(train_sentences)
train_padded = pad_sequences(train_sequences, padding=padding_type, maxlen=max_length)

```


```
validation_sequences = tokenizer.texts_to_sequences(validation_sentences)
validation_padded = pad_sequences(validation_sequences, padding=padding_type, maxlen=max_length)

```


```
label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(labels)

training_label_seq = np.array(label_tokenizer.texts_to_sequences(train_labels))
validation_label_seq = np.array(label_tokenizer.texts_to_sequences(validation_labels))

print(training_label_seq[0])
print(training_label_seq[1])
print(training_label_seq[2])
print(training_label_seq.shape)

print(validation_label_seq[0])
print(validation_label_seq[1])
print(validation_label_seq[2])
print(validation_label_seq.shape)
```

```
model_64_dense = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax')
])
model_64_dense.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model_64_dense.summary()
```


```
num_epochs = 30
history = model_64_dense.fit(train_padded, training_label_seq, epochs=num_epochs, validation_data=(validation_padded, validation_label_seq), verbose=2)
```


```
import matplotlib.pyplot as plt


def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()
  
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")
```


![png](/images/nlp_project_files/nlp_project_10_0.png)



![png](/images/nlp_project_files/nlp_project_10_1.png)



```
model_32_LSTM = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.LSTM(32),
    #tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax')
])
model_32_LSTM.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model_32_LSTM.summary()
```

```
history2 = model_32_LSTM.fit(train_padded, training_label_seq, epochs=60, validation_data=(validation_padded, validation_label_seq), verbose=2)
```


```
plot_graphs(history2, "accuracy")
plot_graphs(history2, "loss")
```


![png](/images/nlp_project_files/nlp_project_13_0.png)



![png](/images/nlp_project_files/nlp_project_13_1.png)



```
model_lstm_conv1d = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv1D(64, 5, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=4),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(6, activation='softmax')
])
model_lstm_conv1d.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model_lstm_conv1d.summary()
```


```
history3 = model_lstm_conv1d.fit(train_padded, training_label_seq, epochs=num_epochs, validation_data=(validation_padded, validation_label_seq), verbose=2)
```

```
plot_graphs(history3, "accuracy")
plot_graphs(history3, "loss")
```


![png](/images/nlp_project_files/nlp_project_16_0.png)



![png](/images/nlp_project_files/nlp_project_16_1.png)



```
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_'+'accuracy'],'--')
plt.plot(history2.history['accuracy'])
plt.plot(history2.history['val_'+'accuracy'],'--')
plt.plot(history3.history['accuracy'])
plt.plot(history3.history['val_'+'accuracy'],'--')
plt.xlabel("Epochs")
plt.ylabel('accuracy')
plt.legend(['ANN_train','ANN_val','LSTM_train','LSTM_val','LSTM_Conv1D_train','LSTM_Conv1D_val'])
plt.show()
```


![png](/images/nlp_project_files/nlp_project_17_0.png)



```
plt.plot(history.history['loss'])
plt.plot(history.history['val_'+'loss'],'--')
plt.plot(history2.history['loss'])
plt.plot(history2.history['val_'+'loss'],'--')
plt.plot(history3.history['loss'])
plt.plot(history3.history['val_'+'loss'],'--')
plt.xlabel("Epochs")
plt.ylabel('loss')
plt.legend(['ANN_train','ANN_val','LSTM_train','LSTM_val','LSTM_Conv1D_train','LSTM_Conv1D_val'])
plt.show()
```


![png](/images/nlp_project_files/nlp_project_18_0.png)



```
x=model_64_dense.layers[0]
weight1=x.get_weights()[0]
y=model_32_LSTM.layers[0]
weight2=y.get_weights()[0]
z=model_lstm_conv1d.layers[0]
weight3=z.get_weights()[0]
print(weight1.shape)
print(weight2.shape)
print(weight3.shape)
```

    (1000, 16)
    (1000, 16)
    (1000, 16)
    


```
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_sentence(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

```


```
from numpy import dot
from numpy.linalg import norm

def cos_sim(a,b):
  return dot(a, b)/(norm(a)*norm(b))

def euc_dist(a,b):
  return np.linalg.norm(a-b)

```


```
import pandas as pd
df=pd.DataFrame(weight1)
```


```
X_corr=df.corr()
values,vectors=np.linalg.eig(X_corr)
eigv_s=(-values).argsort()
vectors=vectors[:,eigv_s]
new_vectors=vectors[:,:2]
new_X=np.dot(weight1,new_vectors)
```


```
import matplotlib.pyplot as plt
vocab_word=list(word_index.keys())
vocab_word=vocab_word[:1000]
random_vocab=np.random.choice(vocab_word,150)
random_index=list(word_index[i] for i in random_vocab)
sampled_X=new_X[random_index,:]
plt.figure(figsize=(13,7))
plt.scatter(sampled_X[:,0],sampled_X[:,1],linewidths=10,color='blue')
plt.xlabel("Dimension 1",size=15)
plt.ylabel("Dimension 2",size=15)
plt.title("Word Embedding Space",size=20)
vocab=len(random_vocab)
for i in range(vocab):
  word=random_vocab[i]
  plt.annotate(word,xy=(sampled_X[i,0],sampled_X[i,1]))
```


![png](/images/nlp_project_files/nlp_project_25_0.png)



