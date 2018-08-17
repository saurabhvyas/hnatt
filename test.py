
# coding: utf-8

# In[1]:


import sys
#from tqdm import tqdm

# In[2]:


""" 
arguments :
first argument: location of test csv , format should be x1(premise) , x2(hypo) , target
see data folder for example csv

second argument: whether to use mode 0  or mode1(prediction each example in testset)
in mode 0, it will only report test set accuracy
in mode 1, it will give output for each example in test set

"""
test_set_file=sys.argv[1]
mode=sys.argv[2]


# In[3]:


#import util.yelp as yelp
import numpy as np
from util.text_util import normalize

from hnatt import HNATT

#YELP_DATA_PATH = 'data/yelp-dataset/yelp.csv'
SAVED_MODEL_DIR = 'saved_models'
SAVED_MODEL_FILENAME = 'model.h5'
EMBEDDINGS_PATH = 'saved_models/glove.6B.100d.txt'

import pandas as pd


# In[ ]:


try:
    print('reading ' + test_set_file + ' .. ')
    df_test=pd.read_csv(test_set_file)
except:
    print('couldnot read csv file')


# In[ ]:


print('''
class labels:
0 : entails
2 : contradiction
1 : neutral
''')


# In[ ]:

data_classes = ["entails", "neutral", "contradiction"]
df_test["x1"] = df_test["x1"] + " " + df_test["x2"]
df_test['target']=df_test['target'].apply(data_classes.index)

print(df_test['target'][:5])


# In[ ]:


df_test=df_test.dropna()


# In[ ]:


col_text = 'x1'
col_target = 'target'


# In[ ]:


y_test = df_test[col_target]


# In[ ]:


def to_one_hot(labels, dim=5):
	results = np.zeros((len(labels), dim))
	for i, label in enumerate(labels):
		results[i][label - 1] = 1
	return results


# In[ ]:


def predict_single(text):
    ntext = normalize(text)
    preds = h.predict([ntext])[0]
    prediction = np.argmax(preds).astype(float)
    return prediction


# In[ ]:


# load pretrained model
try:
    print('loading pretrained model ..')
    h = HNATT()
    h.load_weights(SAVED_MODEL_DIR, SAVED_MODEL_FILENAME)
except:
    print('unable to load pretrained model')


# In[ ]:


if mode=='0':
    print(df_test['x1'][:5])
    df_test['text_tokens']=df_test['x1'].apply(lambda x: normalize(x))

#df['text_tokens'] = df['x1'].progress_apply(lambda x: normalize(x))
#train_set['len'] = train_set['text_tokens'].apply(lambda x: len(x))

    test_x = np.empty((0,))
    test_y = np.empty((0,))

    test_x=df_test['text_tokens']
#train_y=train_set['']



    test_y = to_one_hot(y_test, dim=3)
    print(h.model.metrics_names)
    # test on test set
    loss_and_metrics = h.test(test_x, test_y, batch_size=64)
    print(loss_and_metrics)
    #print(test_x[0])
    #print(test_y[0])
else:
    print('running mode 1 , printing ouput for each test example')
    for row in df_test.itertuples():
        print(row.x1)
        print(predict_single(row.x1))
    
    


# In[ ]:



    



# In[ ]:





# In[ ]:





# In[4]:




