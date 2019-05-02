
# EDA to guide Data Pre-processing


This notebook conducts some EDA to inform some standard standard text data pre-processing steps. We'll be looking at the following analysis:
  - **Relation between proportion of capitalised words to toxicity**: Significant relations means that we can't normalise the case of the words, with corresponding implications for the size of our vocabulary 
  - **Relation between the proportion of stop words to toxicity**: Removing stop words is a standard text analysis pre-processing step. If the proportion of stop words in a sentence gives some indication of toxicity (eg: better written sentences -> more stop words -> less toxic) then we would avoid removing stop words
  - **Relation between presence of punctuation and toxicity**: Removing non-alphanumeric characters is another standard text pre-processing step. If the presence or number of punctaiton in a sentence gives some indication of toxicity, we would avoid undertaking such a pre-processing step

## 0. Loading Data and Dependencies


```python
from src.utils import global_vals as gv
import pandas as pd
from src.featurecreation import count_creation as cc
import nltk
from nltk.corpus import stopwords
```


```python
text_data = pd.read_csv("train.csv")
```


```python
keep_cols_list = ["target", "comment_text", "severe_toxicity", "obscene",
                  "identity_attack", "insult", "threat"]
text_data = text_data[keep_cols_list]
text_data.columns.values
```




    array(['target', 'comment_text', 'severe_toxicity', 'obscene',
           'identity_attack', 'insult', 'threat'], dtype=object)



## 1. Adding feature count and label columns


```python
# ADDING LABEL COLUMN
text_data["label"] = 0
text_data.loc[text_data["target"] >= 0.5, "label"] = 1
```


```python
# Adding counts of words and characters
text_data["pure_wrd_cnt"] = text_data["comment_text"].apply(cc.seq_counter, regex = gv.RGX_PURE_WORD)
text_data["text_len"] = text_data["comment_text"].str.len()
```


```python
# Adding counts of upper case words and characters
text_data["pure_wrd_cnt_upper"] = text_data["comment_text"].apply(cc.seq_counter, regex = gv.RGX_PURE_WORD_UPPER)
text_data["char_cnt_upper"] = text_data["comment_text"].apply(cc.seq_counter, regex = gv.RGX_CHAR_UPPER)
```


```python
# Adding counts general and special non alphanumeric characters
general_punctuation_set = gv.SPL_SEQ_DICT["punctuations"]
special_punctuation_set = gv.SPL_SEQ_DICT["spl_punctuations"]

text_data["gen_punct_cnt"] = text_data["comment_text"].apply(cc.set_seq_counter, 
                                                             set_of_seq = general_punctuation_set)
text_data["spec_punct_cnt"] = text_data["comment_text"].apply(cc.set_seq_counter, 
                                                              set_of_seq = special_punctuation_set)
```


```python
# Adding counts of stop words
stop_words_set = set(stopwords.words('english'))
text_data["stop_word_cnt"] = text_data["comment_text"].apply(cc.set_seq_counter, 
                                                              set_of_seq = stop_words_set)
```

## 2. Normalising count features with length of the comments


```python
text_data["wrd_prop_upper"] = text_data["pure_wrd_cnt_upper"]/text_data["pure_wrd_cnt"]
text_data["gen_punct_prop_upper"] = text_data["gen_punct_cnt"]/text_data["pure_wrd_cnt"]
text_data["spec_punct_prop_upper"] = text_data["spec_punct_cnt"]/text_data["pure_wrd_cnt"]

text_data["char_prop_upper"] = text_data["char_cnt_upper"]/text_data["text_len"]
```

## 3. Checking correlation of toxicity with different pre-proc targets


```python
analysis_df = text_data.loc[:, ["label", "wrd_prop_upper", 
                               "gen_punct_prop_upper", 
                               "spec_punct_prop_upper",
                               "char_prop_upper"]]
```


```python
analysis_df.corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>wrd_prop_upper</th>
      <th>gen_punct_prop_upper</th>
      <th>spec_punct_prop_upper</th>
      <th>char_prop_upper</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>label</th>
      <td>1.000000</td>
      <td>0.006956</td>
      <td>-0.012235</td>
      <td>-0.001607</td>
      <td>-0.000208</td>
    </tr>
    <tr>
      <th>wrd_prop_upper</th>
      <td>0.006956</td>
      <td>1.000000</td>
      <td>0.112306</td>
      <td>0.123979</td>
      <td>0.854858</td>
    </tr>
    <tr>
      <th>gen_punct_prop_upper</th>
      <td>-0.012235</td>
      <td>0.112306</td>
      <td>1.000000</td>
      <td>0.591131</td>
      <td>0.158899</td>
    </tr>
    <tr>
      <th>spec_punct_prop_upper</th>
      <td>-0.001607</td>
      <td>0.123979</td>
      <td>0.591131</td>
      <td>1.000000</td>
      <td>0.157056</td>
    </tr>
    <tr>
      <th>char_prop_upper</th>
      <td>-0.000208</td>
      <td>0.854858</td>
      <td>0.158899</td>
      <td>0.157056</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
summary_df = analysis_df.groupby(["label"], as_index = False).agg({
                                    "wrd_prop_upper": ["mean", "std"],
                                    "gen_punct_prop_upper": ["mean", "std"],
                                    "spec_punct_prop_upper": ["mean", "std"],
                                    "char_prop_upper": ["mean", "std"]})
summary_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>label</th>
      <th colspan="2" halign="left">wrd_prop_upper</th>
      <th colspan="2" halign="left">gen_punct_prop_upper</th>
      <th colspan="2" halign="left">spec_punct_prop_upper</th>
      <th colspan="2" halign="left">char_prop_upper</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th>mean</th>
      <th>std</th>
      <th>mean</th>
      <th>std</th>
      <th>mean</th>
      <th>std</th>
      <th>mean</th>
      <th>std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.015393</td>
      <td>0.059515</td>
      <td>inf</td>
      <td>NaN</td>
      <td>inf</td>
      <td>NaN</td>
      <td>0.037025</td>
      <td>0.048741</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.016943</td>
      <td>0.069912</td>
      <td>inf</td>
      <td>NaN</td>
      <td>inf</td>
      <td>NaN</td>
      <td>0.036987</td>
      <td>0.055641</td>
    </tr>
  </tbody>
</table>
</div>


