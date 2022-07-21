import pandas as pd
from collections import Counter

df = pd.read_csv("detail_dataset_compatible.csv")

# Get unique list of cases
cases = sorted(list(set(df["CASE_ID"])))

# Reduce amount of cases
cases = cases[:int(len(cases)/50)]

# Export reduced dataset
dfReduced = df[df["CASE_ID"].isin(cases)]
dfReduced.to_csv("reduced_Quarter.csv")

# Group by cases
dfGrouped = dfReduced.groupby("CASE_ID")

# How many cases start at day X
casesPerStartDate = Counter(list(dfGrouped["StartDate"].min()))
mostCommonStartDate = sorted(casesPerStartDate.items(), key=lambda x: x[1], reverse=True)[0][0]

print(mostCommonStartDate)  
print(casesPerStartDate[mostCommonStartDate])  

cases = list({name for name, group in dfGrouped if group["StartDate"].min() == mostCommonStartDate})
for i in range(40,200,10):
    dfReduced[dfReduced["CASE_ID"].isin(cases[:i])].to_csv("data_cases_%s.csv" % (str(i)))