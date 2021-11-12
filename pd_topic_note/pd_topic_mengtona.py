#sort_values is mainly to sort the DataFrame in ascending and descending order
import numpy as np
import pandas as pd

#Let's create a DataFrame as df
data = np.random.randn(10, 4)
df = pd.DataFrame(data,columns=['a','b','c','d'])
#view df 
print(df)
#Sort column a of df in ascending order
df.sort_values(by='a')
print(df)
#Sort column a of df in descending order
df.sort_values(by='a',ascending=False)
print(df)
#### 1.2use sort_values to Sort in ascending order
#first,you should determine which column to sort by,second,call sort_values method
#chose a column ,here
df.sort_values(by='a')
print(df)
#or chose b column 
df.sort_values(by='b')
print(df)

#### 1.3use sort_values to Sort in descending order
#Descending order is to add a parameter as 'ascending=False' on the basis of ascending order
#for example,chose a column
df.sort_values(by='a',ascending=False)
print(df)
#or chose b
df.sort_values(by='b',ascending=False)
print(df)
#In fact,'ascending=True'  means ascending,and 'ascending=False'  means descending
#'ascending=True'
df.sort_values(by='a',ascending=True)
print(df)
#'ascending=False'
df.sort_values(by='a',ascending=False)
print(df)

#### 1.4multi-column sorting
#If you want to sort by multiple columns, put all the columns to be arranged into a list, 
#and then assign values to the 'by=' parameter
#for example,wo chose a,b,c columns to sort
df.sort_values(by=['a','b','c'])
print(df)
#Explain，the principle of multi-column sorting is to sort the first column first, 
#then continue to sort the second column on this basis, and so on


#### 1.5ascending and descending order of multi-column sorting
#Sorting overall designation
df.sort_values(by=['a','b','c'])# df sort by first column as a in ascending  order
print(df)
df.sort_values(by=['a','b','c'],ascending=False)## df sort by first column as a in descending order
print(df)

#Sorting freely specified
#The values in 'by=' parameter and 'ascending=' parameter correspond one-to-one


# df  is arranged in descending order in column a first, second in ascending order in column b,and in ascending order in column c last
df.sort_values(by=['a','b','c'],ascending=[False,True,True])
print(df)
# df  is arranged in ascending order in column b first, second in descending order in column c
df.sort_values(by=['b','c'],ascending=[True,False])
print(df)
