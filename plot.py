import pandas as pd
import plotly.express as px

df = pd.read_csv('sleep.csv')

#b=df[(df['Date'] == '22/09/2020') & (df['Time'] > '18:00:00') & (df['Time'] <= '19:00:00')]
#a=b.query('Action in ["Turned Right", "Facing down", "Facing up"]')
#print(len(a))
#fig = px.line(df, x = 'Time', y = 'Action' , title='Driver Monitoring System')
#fig.show()



