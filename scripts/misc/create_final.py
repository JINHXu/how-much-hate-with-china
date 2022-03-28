import pandas as pd
from collections import Counter


def clean_df(df):
    columns = ['geo']
    df.drop(columns=df.columns[:3], 
        axis=1, 
        inplace=True)
    df.drop(columns, inplace=True, axis=1)
    df.rename(columns={'BERT_label': 'COVID_HATE_BERT_PREDS', 'cardiffnlp-roberta-hate-preds':
                       'CARDIFFNLP_ROBERTA_PREDS', 'hateXplain-preds': 'hateXplain_PREDS'}, inplace=True)
    date = []
    for t in df.created_at:
        date.append(t[:10])
    df['date'] = date


# create dfs
q1 = pd.read_csv(
    '/Users/xujinghua/how-much-hate-with-china/predictions/2020/hateXplain/2020-Q1-3.csv')
# correct course
q1_2 = pd.read_csv('/Users/xujinghua/how-much-hate-with-china/predictions/2020/cardiffnlp-roberta-hate/2020-Q1-2.csv')
q1['cardiffnlp-roberta-hate-preds'] = q1_2['cardiffnlp-roberta-hate-preds']

q2 = pd.read_csv(
    '/Users/xujinghua/how-much-hate-with-china/predictions/2020/hateXplain/2020-Q2-3.csv')
q3 = pd.read_csv(
    '/Users/xujinghua/how-much-hate-with-china/predictions/2020/hateXplain/2020-Q3-3.csv')
q4 = pd.read_csv(
    '/Users/xujinghua/how-much-hate-with-china/predictions/2020/hateXplain/2020-Q4-3.csv')


clean_df(q1)
clean_df(q2)
clean_df(q3)
clean_df(q4)

print(Counter(q1['CARDIFFNLP_ROBERTA_PREDS']))
print(Counter(q2['CARDIFFNLP_ROBERTA_PREDS']))
print(Counter(q3['CARDIFFNLP_ROBERTA_PREDS']))
print(Counter(q4['CARDIFFNLP_ROBERTA_PREDS']))


pdList = [q4, q3, q2, q1]
df2020 = pd.concat(pdList)

df2020.to_csv('/Users/xujinghua/how-much-hate-with-china/predictions/2020/FINAL/2020.csv')

print(Counter(df2020['CARDIFFNLP_ROBERTA_PREDS']))


# print(df2020[:100])
# print(df2020[-100:])

# # # test
# # df = pd.read_csv('/Users/xujinghua/how-much-hate-with-china/data/2020/data.csv')
# # print(df)
# # print(len(df2020))