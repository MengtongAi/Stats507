# Stats507
This repo includes what I've down in course STATS 507.

I extract the question 3 in file "HW2-Ai" and put them into a new ipynb document. 

I named the new document as PS2-Question3-Ai and upload it to my repo.

The path of the file PS2-Question3-Ai is: http://localhost:8888/notebooks/Desktop/stats%20507/PS2-Question3-Ai.ipynb

The data in this file I used is coming from National Health and Nutrition Examination Survey NHANES.

I use methods in Python3 to read, clean, and append these data files.

For more details of this: 

1. I use pandas to read and append the demographic datasets keeping only columns containing the unique ids (SEQN), age (RIDAGEYR), race and ethnicity (RIDRETH3), education (DMDEDUC2), and marital status (DMDMARTL), along with the following variables related to the survey weighting: (RIDSTATR, SDMVPSU, SDMVSTRA, WTMEC2YR, WTINT2YR). Then, I add an additional column identifying to which cohort each case belongs, and rename the columns with literate variable names using all lower case and convert each column to an appropriate type. Finally, save the resulting data frame to a serialized “round-trip” format of your choosing (e.g. pickle, feather, or parquet).

2. I repeat what I've down above for the oral health and dentition data (OHXDEN_*.XPT) retaining the following variables: SEQN, OHDDESTS, tooth counts (OHXxxTC), and coronal cavities (OHXxxCTC).

3. At last, I report the number of cases there are in the two datasets above.

As a result, the purpose of this file is to read and clean data. Or we can say, is a way to help you know how to get the data you want from the original huge datasets. 
In ps4 part,I add the gender column and I merge the data by years.

