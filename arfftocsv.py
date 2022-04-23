""" conver arff to csv format"""
import csv
import pandas as pd

def function_arfftocsv(source: str, dest: str = 'processed.csv'):
    """this function deletes @ and empty lines so that produce a no-header csv"""
    fp = open(source)
    rdr = csv.reader(filter(lambda row: row[0]!='@' and len(row)>1, fp))
    with open(dest,'w', newline = '') as csvfile:
        filewriter = csv.writer(csvfile)
        for row in rdr:
            filewriter.writerow(row)
    fp.close()
# this function adds the headers specified in labels argument
def function_labelize(dest: str, labels: list, source: str = 'processed.csv') -> pd.DataFrame:
    """This function takes a destination dir, a source dir, the labels to add
    and returns a dataframe with all labels for each column"""
    df = pd.read_csv(source, names=labels,index_col=False, na_values='?')
    df.to_csv(dest, header=True, index_label=False, index=False)
    return df

def function_dataEncoding(df: pd.DataFrame, labels: list, to_replace: dict, values: dict,
path: str) -> pd.DataFrame:
    """this function encodes explicitly the nominal values of specified labels
    and returns the dataframe with this columns"""
    for label in labels:
        df[label] = df[label].replace(to_replace[label], values[label])
    df[labels].to_csv(path, header= True, index_label= False, index= False)
    return df[labels]

def processing ( all_labels: list,labels: list, to_replace: dict, values: dict,
path: str = 'all.csv', source: str = 'diabetes_paper_fazakis.csv',
 des: str  ='Finaldata.csv')-> pd.DataFrame:
 """this function places the labels for each model and converts categorical to
    numerical data"""
 function_arfftocsv(source)
 df = function_labelize(des, all_labels)
 return function_dataEncoding(df, labels, to_replace, values, path)
 