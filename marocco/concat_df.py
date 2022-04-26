#%%
import sys
from typing import Any, List
import pandas as pd
sys.path.append('C:/Users/panos/Documents/Διπλωματική/code/fz')
from arfftocsv import function_labelize
# %%
def function_concat_df(dest: List[str], labels: List[str],
source: List[str])->pd.DataFrame:
  """
function that takes the dir of a number of csvs, adds column names
 and binds the to one saving them also to 
 specified in source List each of them
"""
  dataframe = pd.DataFrame(columns=labels)
  for d, s in zip(dest, source):
    df = function_labelize(dest=d, labels=labels, source = s)
    dataframe = pd.concat([dataframe, df], axis=0, ignore_index=True)
  return dataframe
# %%
