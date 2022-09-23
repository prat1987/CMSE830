import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns

df = sns.load_dataset("iris")

#Can select the columns from the side bar 
x_lbl = st.sidebar.selectbox('x axis: ', df.columns)
y_lbl = st.sidebar.selectbox('y axis: ', df.drop(columns=[x_lbl]).columns)
z_lbl = st.sidebar.selectbox('z axis: ', df.drop(columns=[x_lbl, y_lbl]).columns)

# Create an object for 3d scatter
Object3d = go.Scatter3d(
    x = df[x_lbl], y = df[y_lbl], z = df[z_lbl],
    mode = 'markers',
    marker = dict(size=5, color='blue'),
)

# Create an object for graph layout
fig = go.Figure(data=[Object3d])
fig.update_layout(
    scene = dict(
    xaxis_title = x_lbl,
    yaxis_title = y_lbl,
    zaxis_title = z_lbl),
    margin=dict(r=30, b=10, l=10, t=10),
)
st.write("""
# Seaborn iris dataset
3D scatter plot between the features for Iris flower  
""")
st.plotly_chart(fig, use_container_width=True)

