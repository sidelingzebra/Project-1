import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

"""Step 1: Data In"""
df=pd.read_csv("Project_1_Data.csv")

#Saving Pandas File as Array A
A=df.to_numpy()
print(A)


"""Step 2: Visalizing Data"""
print(df.info())
fig1=df.hist('X')
fig2=df.hist('Y')
fig3=df.hist('Z')
fig4=df.hist('Step')

"""3D plot creation""""
fig5 = plt.figure()
ax=fig5.add_subplot(projection='3d')

x1=A[:,0]
y1=A[:,1]
z1=A[:,2]

ax.scatter(x1,y1,z1,s=A[:,3],c=A[:,3])

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Location')
ax.legend()
plt.show(fig5)


"""Step 3: Correlation Analysis"""
sb.lmplot(data=df,x="X",y="Step")
sb.lmplot(data=df,x="Y",y="Step")
sb.lmplot(data=df,x="Z",y="Step")

