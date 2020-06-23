import pandas as pd

df = pd.read_csv("processedMS.csv")

df_dict = df.to_dict(orient = "list")

from collections import defaultdict

minNo2idx = defaultdict(list)
for idx, MINERAL_DEPOSIT_NO in enumerate(df_dict["MINERAL_DEPOSIT_NO"]):
    minNo2idx[MINERAL_DEPOSIT_NO].append(idx)

repeatlist = []

for MINERAL_DEPOSIT_NO, idxlist in minNo2idx.items():
    if len(idxlist) > 1:
        repeatlist.append(idxlist[1])


df = df.drop(repeatlist, axis=0)

df = df.loc[:, ['MINERAL_DEPOSIT_NO', 'DEPOSIT_COMMODITY_CODES', 'MAJOR_COMMODITY_CODE', 'MINDEP_CLASS_CODE', "STATUS_TYPE_VALUE", "SIZE_TYPE_VALUE", "ORE_LITHOLOGY_CODE", "LONGITUDE", "LATITUDE"]]
df = df.dropna()
df.to_csv("processedMS.csv")

