from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''check the jupyter notebook one to see better'''

loan_dataset = pd.read_csv('data-t1.csv',low_memory = False)
#loan_dataset.head()
#print(loan_dataset.tail())
#print(loan_dataset.shape)
#print(loan_dataset.info())
#print(loan_dataset.describe())

print(loan_dataset.isnull().sum(axis = 0))

print(loan_dataset['emp_length'].mode())
loan_dataset['emp_length'].fillna(loan_dataset['emp_length'].mode()[0], inplace = True)
loan_dataset['home_ownership'].fillna(loan_dataset['home_ownership'].mode()[0], inplace = True)
print(loan_dataset.isnull().sum(axis = 0))

print(loan_dataset['loan_status'].unique())
print(loan_dataset['home_ownership'].unique())

loan_dataset['emp_length'] = np.where((loan_dataset['home_ownership'].str.contains('years')), loan_dataset['home_ownership'], loan_dataset['emp_length'])
loan_dataset['home_ownership'] = np.where((loan_dataset['annual_inc'].str.contains('MORTGAGE|RENT|OWN|ANY')),loan_dataset['annual_inc'], loan_dataset['home_ownership'])

print(loan_dataset['emp_length'].unique())

loan_dataset.loc[loan_dataset['home_ownership'].str.contains('MOTGAGE|MORGAGE'), 'home_ownership'] = 'MORTGAGE'

loan_dataset.loc[~loan_dataset['home_ownership'].isin(['MORTGAGE','RENT','OWN','ANY']), 'home_ownership'] = loan_dataset['home_ownership'].mode()[0]

print(loan_dataset['home_ownership'].unique())
print(loan_dataset['emp_length'].unique())

print(dict(loan_dataset['emp_length'].value_counts()))

loan_dataset = loan_dataset[loan_dataset.groupby('emp_length').emp_length.transform('count') > 3225]

print(loan_dataset['emp_length'].unique())

loan_dataset.loc[loan_dataset['loan_status'].str.contains('Curren|Curent'), 'loan_status'] = 'Current'
loan_dataset.loc[loan_dataset['loan_status'].str.contains('Fulli Paid|Full Paid'), 'loan_status'] = 'Fully Paid'
loan_dataset.loc[loan_dataset['loan_status'].str.contains('Nov-18|Oct-18|Sep-18|Dec-18|3200'), 'loan_status'] = loan_dataset['loan_status'].mode()[0]

print(loan_dataset['loan_status'].unique())

loan_dataset.loc[loan_dataset['annual_inc'].str.contains('MORTGAGE|RENT|OWN|ANY'), 'annual_inc'] = 0

loan_dataset['term'] = loan_dataset['term'].map(lambda x: x.rstrip('months'))

loan_dataset['term'] = pd.to_numeric(loan_dataset['term'])

loan_dataset['emp_length'] = loan_dataset['emp_length'].map(lambda x: x.rstrip('years'))

loan_dataset['annual_inc'] = pd.to_numeric(loan_dataset['annual_inc'])

print(loan_dataset.dtypes)

def categorizing_data(cell):
    if str(cell) == "10+ ":
        return "> 10"
    elif cell == "< 1 ":
        return "< 1"
    elif cell in ['1 ','2 ', '3 ', '4 ', '5 ']:
        return " 1-5"
    elif cell in ['6 ', '7 ', '8 ', '9 ']:
        return " 6-10"
    else:
        return "0"

loan_dataset['emp_length'] = loan_dataset.apply(lambda cell: categorizing_data(cell.emp_length), axis=1)

print(loan_dataset['emp_length'].unique())

Q1 = loan_dataset.quantile(0.25)
Q3 = loan_dataset.quantile(0.75)
IQR = Q3 - Q1
print(IQR)

loan_dataset = loan_dataset[~((loan_dataset < (Q1 - 1.5 * IQR)) |(loan_dataset > (Q3 + 1.5 * IQR))).any(axis=1)]
loan_dataset.shape

fig = plt.figure()
plt.boxplot(loan_dataset['loan_amnt'])
plt.show

loan_dataset.to_csv('cleaned.csv', index = None, header = True)
