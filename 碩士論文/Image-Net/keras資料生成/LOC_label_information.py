# 將labek_information的WNID&class_name抓出來(才符合順序)

import pandas as pd
df=pd.read_excel('../label_information.xlsx')
df=df[['WNID','class_name']]
print(df)
df=df.values.tolist()

with open('./LOC_label_information.txt','a',encoding='utf-8') as wf:
    for row in df:
        str_row=""
        for col in row:
            col=col.strip()     #清空格
            str_row=str_row+col+" "
        str_row=str_row.strip()
        wf.write(str_row+"\n")