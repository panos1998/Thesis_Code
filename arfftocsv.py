from operator import index
import numpy as np
import pandas as pd
import csv
# this function deletes @ and empty lines so that produce a no-header csv
def arfftocsv(source: str, dest: str = 'processed.csv'):
    fp = open(source)
    rdr = csv.reader(filter(lambda row: row[0]!='@' and len(row)>1, fp))
    with open(dest,'w', newline = '') as csvfile:
         filewriter = csv.writer(csvfile)
         for row in rdr:
            filewriter.writerow(row)
    fp.close()
# this function adds the headers specified in labels argument
def labelize(dest: str, labels: list, source: str = 'processed.csv') -> pd.DataFrame:
    df = pd.read_csv(source, names=labels,index_col=False, na_values='?')
    df.to_csv(dest, header=True, index_label=False, index=False)
    return df

#  this function encodes explicitly the nominal values of specified labels and returns the dataframe with this columns
def dataEncoding(df: pd.DataFrame, labels: list, to_replace: dict, values: dict, path: str) -> pd.DataFrame:
    for label in labels:
        df[label] = df[label].replace(to_replace[label], values[label])
    df[labels].to_csv(path, header= True, index_label= False, index= False)
    return df[labels]

# this function places the labels for each model and converts categorical to numerical data
def processing ( all_labels: list,labels: list, to_replace: dict, values: dict, path: str = 'all.csv', source: str = 'diabetes_paper_fazakis.csv',
 des: str  ='Finaldata.csv')-> pd.DataFrame:
 arfftocsv(source)
 df = labelize(des, all_labels)
 return dataEncoding(df, labels, to_replace, values, path)
 
 