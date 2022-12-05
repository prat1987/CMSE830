import streamlit as st
import seaborn as sns
import pandas as pd
import altair as alt
import numpy as np
import matplotlib.pyplot as plt
import hiplot as hip
from prediction_ml import DataProcess
#Python Machine learning package
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
from sklearn.metrics import mean_squared_error
# get data
#df_KBS = sns.load_dataset("mpg")
df_KBS = pd.read_csv("kbs_final_data.csv")
df_KBS['Management']=df_KBS['Treatment1'].str[:2]
# st.write(df_KBS)

labels_mpg = df_KBS.select_dtypes(include='float64').columns  # feel free to change this
# st.write(labels_mpg)
def correlation(df,col1,col2):
    rmse=np.sqrt(mean_squared_error(df[col1], df[col2]))
    r=stats.pearsonr(df[col1], df[col2])
    #r_square=r2_score(ARL_R['prediction.ARL'], ARL_R['N2O'])
    r_square= pow(r[0], 2)
    return rmse,r,r_square


#######################
## streamlit sidebar ##
#######################

st.sidebar.title("""
# Please select the variable from a dropdown menu to plot 
""")
st.write("""
# Kellogs Biological Station (KBS) Nitrous oxide and other variables dataset  
""")

st.write('N2O is one of the major greenhouse gases with a global warming potential (GWP) of 273 (± 118) times \
    more than carbon dioxide (CO2) for a 100-year time scale.This web app will let the user \
    explore the relationship between several variables measured for a experiment which will help them to \
    understand the factors impacting greenhouse gas (GHG) emissions from agricultural soils'
)

st.header("Visualization")

selection=st.sidebar.selectbox(
    "Please select one option", #Drop Down Menu Name

    [
        "Data analysis",
        "Run ML Algorithm", #First option in menu
       
    ]
)
if (selection=='Data analysis'):

    sd = st.sidebar.selectbox(
        "Select a Plot", #Drop Down Menu Name

        [
            "Hi Plot",
            "Violin Plot", #First option in menu
            "Strip Plot" ,  #Seconf option in menu
            "Dist Plot" ,
        #     "Joint Plot"
        ]
    )


    # allow user to choose which portion of the data to explore


    fig = plt.figure(figsize=(12, 6))

    if sd =='Hi Plot':
        # just convert it to a streamlit component with `.to_streamlit()` before
        xp=hip.Experiment.from_dataframe(df_KBS[labels_mpg])
        ret_val = xp.to_streamlit( key="hip").display()

        #st.markdown("hiplot returned " + json.dumps(ret_val))
        
    elif sd == "Violin Plot":
        x_axis_choice = st.sidebar.selectbox(
        "x axis",
        labels_mpg)
        y_axis_choice = st.sidebar.selectbox(
        "y axis",
        df_KBS.columns,index= len(df_KBS.columns.tolist())-1)
        #hue_in = st.sidebar.selectbox('hue: ', df_KBS.columns)
        sns.violinplot(x = x_axis_choice, y = y_axis_choice, data =df_KBS)

    elif sd == "Strip Plot":
        x_axis_choice = st.sidebar.selectbox(
        "x axis",
        labels_mpg)
        y_axis_choice = st.sidebar.selectbox(
        "y axis",
        df_KBS.columns,index= len(df_KBS.columns.tolist())-1)
        sns.stripplot(x = x_axis_choice, y = y_axis_choice, data = df_KBS)
    elif sd == "Dist Plot":
        #sns.distplot(df ,x=x_lbl, y=y_lbl, rug=True, kind='kde',hue=hue_in)
        x_axis_choice = st.sidebar.selectbox(
        "x axis",
        labels_mpg)
        y_axis_choice = st.sidebar.selectbox(
        "y axis",
        labels_mpg)
        hue_in = st.sidebar.selectbox('hue: ', df_KBS.columns,index= len(df_KBS.columns.tolist())-1)
        sns.displot(df_KBS, x=x_axis_choice, y=y_axis_choice, hue=hue_in, kind="kde",rug=True)

    # elif sd =="Joint Plot":
    #     x_axis_choice = st.sidebar.selectbox(
    #     "x axis",
    #     labels_mpg)
    #     y_axis_choice = st.sidebar.selectbox(
    #     "y axis",
    #     labels_mpg)
    #     sns.jointplot(x = x_axis_choice, y = y_axis_choice, data = df_KBS, color="#7f1a1a")

        #sns.displot(df_mpg, x="horsepower", y="mpg", kind="kde", rug=True)
    st.pyplot(fig)
elif (selection=='Run ML Algorithm'):
    # Classify the target values into different bins and labels  
    bins = [-5, 1, 5, 10, 15, 20, 35, 60, 600]

    labels = [
            "verylow", "mediumlow", "low", "low-medium", "medium",
            "medium-high", "high", "extreme"
        ]
    # Initialize the DataProcess class to finalize the data to input into Machine learning algorithm
    dp=DataProcess(df_KBS)

    #Drop the features
    
    options = st.sidebar.multiselect( 'Columns to drop', df_KBS.columns.tolist(),
    ['Date_found_N',
    'Date_found_NDVI',
    'Date_found_W',
    'Date_found_leach',])
 
    st.write('You selected:', options)
    df_filter=dp.drop_columns(options)
    #Use Label encoder to convert object or string to numeric values
    df_encode=dp.label_enc()
    size_test= st.sidebar.slider(label='Test size', key='Size',value=0.3, min_value=0.1, max_value=0.9, step=0.1)
    # Split the data into training and testing 
    Xtrain, Xtest, Ytrain, Ytest = dp.split_data('N20',testsize=size_test,strat=True,bins=bins,labels=labels)

    important = st.multiselect(
    'Select features model building ',
    df_KBS.columns.tolist(),
    [ 'Ammonium',  'SWC', 'Nitrate','NDVI'])

        # Define the regression model that you want to use 
    reg = RandomForestRegressor(n_estimators = 800, random_state = 42,max_features=4,bootstrap=True)

    #Intialize the MachineLearningalgo class to perform final task 
    #MM=MachineLearningalgo(Xtrain, Xtest, Ytrain, Ytest,important,reg)

    #visualizer = PredictionError(reg)

    # Fit the training data to the visualizer
    reg.fit(Xtrain[important], Ytrain['N20'])

# Use the forest's predict method on the test data
    predictions = reg.predict(Xtest[important])
    df_mpg=pd.DataFrame()
    df_mpg=Xtest[important]
    df_mpg['Predicted']=predictions
    df_mpg['Observed']=Ytest['N20']
    scatter = alt.Chart(df_mpg).mark_circle(size=100).encode(x='Predicted', y='Observed',
    tooltip=['Ammonium', 'Nitrate','NDVI']).interactive()
    #MM.visulaization_error()
    rmse_cal,r_cal,r_square_cal=correlation(df_mpg, 'Predicted','Observed')
    st.write("R${^{2}}$ = ",np.round(r_square_cal,2))
    st.write("R = ",np.round(r_cal[0],2))
    st.write("RMSE = ",np.round(rmse_cal,2))


    scatter
