
# coding: utf-8

# In[1]:


import util.yelp as yelp
import numpy as np
from util.text_util import normalize

from hnatt import HNATT

#YELP_DATA_PATH = 'data/yelp-dataset/yelp.csv'
SAVED_MODEL_DIR = 'saved_models'
SAVED_MODEL_FILENAME = 'model.h5'
EMBEDDINGS_PATH = 'saved_models/glove.6B.100d.txt'

import pandas as pd


# In[2]:


from keras import backend as K
print(K.tensorflow_backend._get_available_gpus())


# In[3]:


df=pd.read_csv('data/preprocessing/preprocessing_scripts/output/concatenated_train_pandas.csv')
df_valid=pd.read_csv('data/preprocessing/preprocessing_scripts/output/concatenated_dev_pandas.csv')
df_test=pd.read_csv('data/concatenated_test_pandas.csv')


# In[4]:


#print(df["x1"][0])
#print(df["x2"][0])


# In[5]:


df["x1"] = df["x1"] + " " + df["x2"]
df_valid["x1"] = df_valid["x1"] + " " + df_valid["x2"]
df_test["x1"] = df_test["x1"] + " " + df_test["x2"]


# In[6]:


#print(df["x1"][5])


# In[7]:


#print(df_test['x1'][6])


# In[8]:


#print(df_valid['target'])


# In[9]:


#for row in df_test.itertuples():
 #   print(row.x1)


# In[10]:


# convert class labels into class indices
data_classes = ["entails", "neutral", "contradiction"]
df['target']=df['target'].apply(data_classes.index)
df_valid['target']=df_valid['target'].apply(data_classes.index)
df_test['target']=df_test['target'].apply(data_classes.index)


# In[11]:


#print(df_test['target'][:15])


# In[12]:


df=df.dropna()
df_valid=df_valid.dropna()
df_test=df_test.dropna()


# In[13]:


#df=df[:int(df.shape[0]*0.001)]
#df_valid=df_valid[:int(df_valid.shape[0]*0.001)]
#df_test=df_test[:int(df_test.shape[0]*0.001)]


# In[14]:


col_text = 'x1'
col_target = 'target'


# In[15]:


y_train = df[col_target]
y_test = df_test[col_target]
y_val = df_valid[col_target]


# In[16]:


#print(y_val[4])


# In[17]:


#train_x=df["x1"]
#train_y=y_train


# In[18]:


#train_y.shape


# In[19]:


'''
h = HNATT()	
h.train(train_x, train_y, 
batch_size=16,
epochs=16,
embeddings_path=None)

#h.load_weights(SAVED_MODEL_DIR, SAVED_MODEL_FILENAME)



# print attention activation maps across sentences and words per sentence
activation_maps = h.activation_maps(
'they have some pretty interesting things here. i will definitely go back again.')
print(activation_maps)
'''


# In[20]:


#(train_x, train_y), (test_x, test_y) = yelp.load_data(path='yelp.csv', size=1e4)


# In[21]:


def to_one_hot(labels, dim=5):
	results = np.zeros((len(labels), dim))
	for i, label in enumerate(labels):
		results[i][label - 1] = 1
	return results


# In[22]:


print('loading training set ...')
df['text_tokens'] = df['x1'].progress_apply(lambda x: normalize(x))
#train_set['len'] = train_set['text_tokens'].apply(lambda x: len(x))

train_x = np.empty((0,))
train_y = np.empty((0,))

training_len=df['x1'].shape[0]

#train_set = df[0:training_len].copy()
#train_set['len'] = train_set['text_tokens'].apply(lambda x: len(x))

train_x=df['text_tokens']
#train_y=train_set['']



train_y = to_one_hot(y_train, dim=3)
print(train_x[0])
print(train_y[0])

#test_y = to_one_hot(test_y)


# In[23]:


# preprocess test_x , as above block
print('loading test set ...')
df_test['text_tokens']=df_test['x1'].progress_apply(lambda x: normalize(x))

#df['text_tokens'] = df['x1'].progress_apply(lambda x: normalize(x))
#train_set['len'] = train_set['text_tokens'].apply(lambda x: len(x))

test_x = np.empty((0,))
test_y = np.empty((0,))

test_x=df_test['text_tokens']
#train_y=train_set['']



test_y = to_one_hot(y_test, dim=3)
print(test_x[0])
print(test_y[0])



# In[24]:


#preprocess val_x , as above block
print('loading validation set ...')

df_valid['text_tokens']=df_valid['x1'].progress_apply(lambda x: normalize(x))

#df['text_tokens'] = df['x1'].progress_apply(lambda x: normalize(x))
#train_set['len'] = train_set['text_tokens'].apply(lambda x: len(x))

valid_x = np.empty((0,))
valid_y = np.empty((0,))

valid_x=df_valid['text_tokens']
#train_y=train_set['']



valid_y = to_one_hot(y_val, dim=3)
print(valid_x[0])
print(valid_y[0])


# In[25]:


# load pretrained model
print('loading pretrained model / restoring model ...')
h = HNATT()
h.load_weights(SAVED_MODEL_DIR, SAVED_MODEL_FILENAME)


# In[38]:


# test on test set
#loss_and_metrics = h.test(test_x, test_y, batch_size=64)
#print(loss_and_metrics)


# In[39]:


#h.model.metrics_names


# In[45]:


#valid_y[0]


# In[26]:


#h = HNATT()	
h.train(train_x, train_y,valid_x,valid_y,
batch_size=64,
epochs=16,
embeddings_path=None,saved_model_dir=SAVED_MODEL_DIR,
saved_model_filename='mnli.h5')

#h.load_weights(SAVED_MODEL_DIR, SAVED_MODEL_FILENAME)



# print attention activation maps across sentences and words per sentence
#activation_maps = h.activation_maps(
#'they have some pretty interesting things here. i will definitely go back again.')
#print(activation_maps)


# In[ ]:


''' 
activation_maps = h.activation_maps(
'they have some pretty interesting things here. i will definitely go back again.')
print(activation_maps)
'''


# In[24]:


#train_y[0]


# In[23]:


#h.model.summary()


# In[9]:


'''
text='he was not well, he stayed at home only. he didnt go to office '

ntext = normalize(text)
preds = h.predict([ntext])[0]
prediction = np.argmax(preds).astype(float)
print(prediction)
'''


# In[ ]:


#he is dancing with joy because its his birthday. he is very happy
# 

