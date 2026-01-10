import pandas as pd

data = {
    '部门': ['技术部', '市场部', '技术部', '市场部', '宠物部'],
    '工资': [10000, 8000, 12000, 9000, 500],
    '工号': [1, 2, 3, 4, 5]
}
df = pd.DataFrame(data)
print(df)
print(df.groupby('部门')['工资'].mean().reset_index())
print(df.groupby('部门')['工资'].agg(['mean', 'max', 'min']).reset_index())

#自定义列名
resultdf = df.groupby('部门').agg(人数=('工号', 'count'),平均工资=('工资', 'mean'), 最高工资=('工资', 'max'), 最低工资=('工资', 'min')).sort_values(by=['人数','平均工资'],ascending=[False,True]).reset_index()
resultdf.to_csv('rs01.csv',index=False)
print(resultdf)