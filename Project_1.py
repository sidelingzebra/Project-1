import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sb

"""Step 1: Data In"""
df=pd.read_csv("Project_1_Data.csv")


"""Step 2: Visalizing Data"""
# print(df.info())
# fig1=df.hist('X')
# fig2=df.hist('Y')
# fig3=df.hist('Z')
# fig4=df.hist('Step')


"""Step 3: Correlation Analysis"""
sb.lmplot(data=df,x="X",y="Step")
sb.lmplot(data=df,x="Y",y="Step")
sb.lmplot(data=df,x="Z",y="Step")

