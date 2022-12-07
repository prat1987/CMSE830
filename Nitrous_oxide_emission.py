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
from xgboost.sklearn import XGBRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
import streamlit.components.v1 as components
from scipy import stats
#from pycaret.regression import *
import math
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
#from missingpy import MissForest
import sys
#import sklearn.neighbors._base
import base64
#sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
# get data
#df_KBS = sns.load_dataset("mpg")
#df_KBS = pd.read_csv("kbs_final_data.csv")
#df_KBS = pd.read_csv("kbs_final_data_salus_evi_soiltext.csv")
#tab1, tab2, tab3 = st.tabs(["Cat", "Dog", "Owl"])
def Null_testing(df_KBS):
    missing=False
    if df_KBS.isnull().values.any():
                missing=True
                st.write('### Error: Data contain missing values')
                for column in df_KBS.columns.tolist():
                    if df_KBS[column].isnull().sum(axis = 0)>0:
                        st.write("Feature", column, "with Nan values within dataset")
                        #st.write(column)
                        st.write("Total NaN values:",df_KBS[column].isnull().sum(axis = 0))
    return missing     

def category_columns(df):
        '''
        The function is used to  encode the string columns of the input dataframe 
        into numeric to prepare it to feed in ML model.

        Returns:
        
            dataframe: A dataframe with encoded columns
        '''
        list_columns=[]
        for each in df.columns.values:
            encode = ['category', 'object', 'string']
            if df[each].dtypes in (encode):
                 list_columns.append(each)
                
        return list_columns             
    
st.write("""
    # Tool for data analysis and run Machine learning models 
    """)



st.write('N2O is one of the major greenhouse gases with a global warming potential (GWP) of 273 (± 118) times \
    more than carbon dioxide (CO2) for a 100-year time scale. This web app will let the user \
    explore the relationship between several variables measured and N2O emissions for a experiment using visualization and ML models which will help them to \
    understand the major factors impacting greenhouse gas (GHG) emissions from agricultural soils.'
)
st.write("""Author: Prateek Sharma""")
font_css = """
<style>
button[data-baseweb="tab"] {
  font-size: 26px;
}
</style>
"""
st.write(font_css, unsafe_allow_html=True)
whitespace=9
listTabs = ["Background", "Dataset", "Tool", "Author"]
#tab1, tab2, tab3, tab4 = st.tabs(["Background", "Dataset", "Tool", "Author"])
tab1, tab2, tab3, tab4 = st.tabs([s.center(whitespace,"\u2001") for s in listTabs])

with tab1:
    st.write("""

    ### Background: Nitrous oxide (N2O)
    """)
    st.image('./n2o_emission_copy.jpg')
    st.write('N2O emissions from soils are primarily \
    produced during two microbial-driven biological processes,nitrification and denitrification.\
    Nitrification is the aerobic microbial oxidation, in \
    which ammonium ion is oxidized into nitrate and N2O is released as a byproduct.\
    n the other hand, denitrification is \
    the anaerobic microbial (mainly bacterial) \
    reduction of nitrate to nitrite and then to the gasses \
    NO, N2O, and N2. N2O production depends on the amount of mineral N substrates in the soil, i.e., ammonium and nitrate. Therefore additions of mineral N \
    fertilizers and other sources of N (manures, residue) to agricultural soil \
    are considered the primary drivers of N2O emissions and higher atmospheric .')
    
    st.write("""

    ##### Global budget Nitrous oxide (N2O)
    """)
    st.image('./N2O_budget.jpg')
    st.markdown('**Source: Global nitrous oxide budget 2007-16. Adopted from Tian et al  2020.**')
    link = '[Tian et al. 2020](https://www.nature.com/articles/s41586-020-2780-0)'
    st.markdown(link, unsafe_allow_html=True)

    st.write('Nitrous Oxide (N2O) is a highly potent greenhouse gas (GHG), \
    with a global warming potential (GWP) of 298 that of CO2 on a 100-year timescale \
    (EPA, 2018). The concentration of atmospheric N2O has increased \
    by more than 20%, from 270 parts per billion (ppb) in 1750 to 331 ppb \
    in 2018 (Tian et al., 2020). This dramatic shift is primarily driven by \
    increased anthropogenic sources that raise current total global emissions \
    to ~17 teragram (Tg) nitrogen (N) (Syakila and Kroeze et al., 2011; Thompson et al., 2019; Tian et al., 2020).\
    Approximately 52% of anthropogenic emissions come from the direct emissions from Nitrogen (N) addition in the agricultural sector (Tian et al., 2020)')

with tab2:
    import base64
    def show_pdf(file_path):
        with open(file_path,"rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="800" height="800" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)

    
    st.write("""

    ### Dataset: Kellogs Biological Station (KBS) experimental nitrous oxide and other variables dataset
    """)
    st.image('./current-kbs-lter-msce-plot-map.jpg')
    #show_pdf('./current-kbs-lter-msce-plot-map.pdf')

    st.write("""

    ##### Experimental site data:
    """)
    st.write("Field observations collected at the LTER MCSE site are stored on the KBS website. \
        Data include N2O gas emissions, which have been measured since 1991 \
        at varying temporal frequency (monthly or biweekly) using permanently installed in-situ static \
        chambers). Predictor variables include soil inorganic N (ammonium [NH4 +-N] and nitrate [NO3 –N]), \
        soil water content (SWC), and bulk density (BD). We used measured soil inorganic N data \
        interpolated between the soil sampling dates to match the manual N!O gas sampling dates. \
        The water-filled pore space (WFPS), a proxy for soil O2 limitation, was estimated from volumetric \
        water content (VWC) and BD in the top 25 cm soil layer. VWC was calculated from BD and \
        gravimetric soil moisture values measured at each manual gas sampling event")
    

    
    st.markdown('**HELP: Please click on the following link for more information about the dataset used in this tool**')
    link = '[MCSE KBS LTER experiment](https://lter.kbs.msu.edu/datatables)'
    st.markdown(link, unsafe_allow_html=True)
with tab3:
    Exp_file_opt = st.radio(
                    "Please select the option for the experiment file",
                    ('Upload a file', 'Run the default'))
    flag=0
    #missing=False
    #flag_default=1
    if Exp_file_opt=='Upload a file':
        
        flag=1
        file_format = st.radio(
                    "Please select the format for the experiment file",
                    ('CSV', 'Excel'))

        uploaded_file = st.file_uploader("Please provide an experiment file for analysis")
    #df_KBS = pd.read_csv("kbs_final_data_interpolate_evi.csv")
        if uploaded_file is not None:
            
            if (file_format=='CSV'):
                df_KBS = pd.read_csv(uploaded_file)
                
            elif(file_format=='Excel'):
                st.write(uploaded_file)
                df_KBS = pd.read_excel(open('Saha_et_al_2020_ERL_Data.xlsx','rb'),sheet_name='Data')
            #st.write(df_KBS.isnull().sum())
            missing=Null_testing(df_KBS)
            if (missing ==True):
                with st.form(key='Missing'):
                    Miss_opt = st.radio(
                    "Please select the option for dealing with missing values",
                    ('Drop', 'Fill')) 
                    submit_button_miss=st.form_submit_button(label='Submit')
                if submit_button_miss:

                    if Miss_opt=='Drop':
                        df_KBS=df_KBS.dropna(axis=0)
                        df_KBS=df_KBS.reset_index(drop=True)
                        missing=Null_testing(df_KBS)
                        if (missing==False):
                            st.write("### Missing values dropped")
                        else:
                            st.write("### Still missing values exists")

                    

                        
                else:
                    labels_mpg = df_KBS.select_dtypes(include='float64').columns.tolist()
                    label_cat=category_columns(df_KBS)
                    imputer = MissForest(random_state = 500)
                    df_imputer=imputer.fit_transform(df_KBS[labels_mpg])
                    df_impute = pd.DataFrame(data = df_imputer,    
                                                columns =labels_mpg )
                    df_impute_com=pd.concat([df_impute,df_KBS[label_cat]],axis=1)
                    missing=Null_testing(df_impute_com)
                    if (missing==False):
                        st.write("### Missing values Filled using MissForest imputer ")
                        df_KBS=df_impute_com
                    else:
                        st.write("### Still missing values exists")
            

               
            # if df_KBS.isnull().values.any():
            #     missing=True
            #     st.write('### Error: Data contain missing values')
            #     for column in df_KBS.columns.tolist():
            #         if df_KBS[column].isnull().sum(axis = 0)>0:
            #             st.write("Feature", column, "with Nan values within dataset")
            #             #st.write(column)
            #             st.write("Total NaN values:",df_KBS[column].isnull().sum(axis = 0))
    else:
        df_KBS = pd.read_csv("kbs_final_data_interpolate_evi.csv")
        flag=2
    # else:
    #     st.write(" ##Warning!!!Please select one of the two option for experiment files")
    #     df_KBS=pd.DataFrame(data=None)
    if ((flag==1 and  uploaded_file is not None and missing==False) or flag==2):
        if flag ==2:
            df_KBS['Management']=df_KBS['Treatment1'].str[:2]
            df_KBS=df_KBS[(df_KBS['Ammonium']>0.) & (df_KBS['Nitrate']>0.)]
            # st.write(df_KBS)
            bins = [-5, 1, 5, 10, 15, 20, 35, 60, 600]

            labels = [
                    "verylow", "mediumlow", "low", "low-medium", "medium",
                    "medium-high", "high", "extreme"
                ]
            # Initialize the DataProcess class to finalize the data to input into Machine learning algorithm
            #dnew=DataProcess(df_KBS)
            #dnew.bin_prep(bins,labels)

        labels_mpg = df_KBS.select_dtypes(include='float64').columns.tolist()  # feel free to change this
        # st.write(labels_mpg)
        def correlation(df,col1,col2):
            rmse=np.sqrt(mean_squared_error(df[col1], df[col2]))
            r=stats.pearsonr(df[col1], df[col2])
            #r_square=r2_score(ARL_R['prediction.ARL'], ARL_R['N2O'])
            r_square= pow(r[0], 2)
            return rmse,r,r_square
        @st.cache(allow_output_mutation=True)
        def exp_data():
            return []
        def feature_important_tree(regmodel,names,model_type):
            all_feat_imp_df = pd.DataFrame(data=[tree.feature_importances_ for tree in 
                                            regmodel],
                                    columns=names)

            fig, ax = plt.subplots(figsize=(15, 5))
            (sns.boxplot(data=all_feat_imp_df,ax=ax)
                    .set(title=model_type+' '+'Feature Importance Distributions',
                        ylabel='Importance'))
            #fig.savefig('../../OUTPUT_FILES/ESMRC/SALUS/'+filename,dpi=300)

        #######################
        ## streamlit sidebar ##
        #######################

        st.sidebar.title("""
        Please select option for data anlysis or runnign ML model from a dropdown menu 
        """)
        

        #st.header("Visualization")
        #st.
        selection=st.sidebar.selectbox(
            "Please select one option", #Drop Down Menu Name

            [
                "Data analysis",
                "Run ML Algorithm", #First option in menu
            
            ]
        )
        if (selection=='Data analysis'):
            with st.form(key='Plot'):
                with st.sidebar:
                    sd = st.sidebar.selectbox(
                        "Select a Plot", #Drop Down Menu Name

                        [
                            "Hi Plot",
                            'Pairplot',
                            "Violin Plot", #First option in menu
                            "box_plot" ,  #Seconf option in menu
                            "Dist Plot" ,
                            "lm Plot",
                            "Joint Plot",
                            "Line Plot"

                        ]
                    )
                    submit_button_plot=st.form_submit_button(label='Plot')


            # allow user to choose which portion of the data to explore
            if sd in ['Hi Plot','Pairplot']:
                if (flag==2):
                    features_sel = st.sidebar.multiselect( 'Select features to plot', df_KBS.columns.tolist(),
                    ['Management','Ammonium',  'WFPS', 'Nitrate','NDVI'])
                else:
                    features_sel = st.sidebar.multiselect( 'Select features to plot', df_KBS.columns.tolist(),
                    labels_mpg)
                if sd=='Pairplot':
                    hue_in = st.sidebar.selectbox('hue: ', features_sel,index=0)
            elif sd in ["Violin Plot", "box_plot"]:
                x_axis_choice = st.sidebar.selectbox(
                "x axis",
                labels_mpg)
                y_axis_choice = st.sidebar.selectbox(
                "y axis",
                df_KBS.columns,index= len(df_KBS.columns.tolist())-1)
            elif sd in ["Dist Plot", "lm Plot"]:
                x_axis_choice = st.sidebar.selectbox(
                "x axis",
                labels_mpg)
                y_axis_choice = st.sidebar.selectbox(
                "y axis",
                labels_mpg,index=1)
                if (flag==2):
                    hue_in = st.sidebar.selectbox('hue: ', ['Crop','Management','Stability','bins'])
                else:
                    hue_in = st.sidebar.selectbox('hue: ', df_KBS.columns.tolist())
            elif sd in ["Joint Plot", "Line Plot"]:
                y_axis_choice = st.sidebar.selectbox(
                "y axis",
                labels_mpg)
                if sd =="Line Plot":
                    x_axis_choice = st.sidebar.selectbox(
                    "x axis",
                    df_KBS.columns.tolist())
                else:
                    x_axis_choice = st.sidebar.selectbox(
                    "x axis",
                    labels_mpg)
                    


            fig = plt.figure(figsize=(12, 6))
            if submit_button_plot:
                if sd =='Hi Plot':
                    # # just convert it to a streamlit component with `.to_streamlit()` before
                    # if (flag==2):
                    #     features_sel = st.sidebar.multiselect( 'Select features to plot parallel pairplot', df_KBS.columns.tolist(),
                    #     ['N20','Ammonium',  'WFPS', 'Nitrate','NDVI'])
                    # else:
                    #     features_sel = st.sidebar.multiselect( 'Select features to plot parallel plot', df_KBS.columns.tolist(),labels_mpg)
                    # xp=hip.Experiment.from_dataframe(df_KBS[features_sel])
                    xp=hip.Experiment.from_dataframe(df_KBS[features_sel])
                    ret_val = xp.to_streamlit( key="hip").display()

                    #st.markdown("hiplot returned " + json.dumps(ret_val))
                    
                elif sd=='Pairplot':
                    #fig_all = plt.figure()
                    # if (flag==2):
                    #     features_sel = st.sidebar.multiselect( 'Select features to plot pairplot', df_KBS.columns.tolist(),
                    #     ['Management','Ammonium',  'WFPS', 'Nitrate','NDVI'])
                    # else:
                    #     features_sel = st.sidebar.multiselect( 'Select features to plot pairplot', df_KBS.columns.tolist(),
                    #     labels_mpg)
                    
                    #labels_mpg.append('Management')
                    #hue_in = st.sidebar.selectbox('hue: ', features_sel,index=0)
                    fig_all=sns.pairplot(df_KBS[features_sel], hue=hue_in) 
                    st.pyplot(fig_all)

                elif sd == "Violin Plot":
                    # x_axis_choice = st.sidebar.selectbox(
                    # "x axis",
                    # labels_mpg)
                    # y_axis_choice = st.sidebar.selectbox(
                    # "y axis",
                    # df_KBS.columns,index= len(df_KBS.columns.tolist())-1)
                    #hue_in = st.sidebar.selectbox('hue: ', df_KBS.columns)
                
                    ax=sns.violinplot(x = x_axis_choice, y = y_axis_choice, data =df_KBS)

                    plt.ylabel(y_axis_choice, fontsize=20)
                    plt.xlabel(x_axis_choice, fontsize=20)
                    plt.tick_params(axis='both', labelsize=18)
                    plt.legend(loc=(1,.4),fontsize=14,frameon=False)
                    st.pyplot(fig)
                elif sd == "box_plot":
                    # x_axis_choice = st.sidebar.selectbox(
                    # "x axis",
                    # labels_mpg)
                    # y_axis_choice = st.sidebar.selectbox(
                    # "y axis",
                    # df_KBS.columns,index= len(df_KBS.columns.tolist())-1)
                    #interactive=st.sidebar.checkbox('interactive')
                    
                    box=alt.Chart(df_KBS, width=300, height=300).mark_boxplot().encode(y=y_axis_choice, x=x_axis_choice)
                    box
                #   x_axis_choice = st.sidebar.selectbox(
                #     "x axis",
                #     labels_mpg)
                #     y_axis_choice = st.sidebar.selectbox(
                #     "y axis",
                #     df_KBS.columns,index=2)
                #     sns.stripplot(x = x_axis_choice, y = y_axis_choice, data = df_KBS)
                #     plt.ylabel(y_axis_choice, fontsize=14)
                #     plt.xlabel(x_axis_choice, fontsize=14)
                #     plt.tick_params(axis='both', labelsize=12)
                #     plt.legend(loc=(1,.4),fontsize=14,frameon=False)
                #     st.pyplot(fig)
                elif sd == "Dist Plot":
                    #sns.distplot(df ,x=x_lbl, y=y_lbl, rug=True, kind='kde',hue=hue_in)
                    # x_axis_choice = st.sidebar.selectbox(
                    # "x axis",
                    # labels_mpg)
                    # y_axis_choice = st.sidebar.selectbox(
                    # "y axis",
                    # labels_mpg,index=1)
                    # if (flag==2):
                    #     hue_in = st.sidebar.selectbox('hue: ', ['Crop','Management','Stability','bins'])
                    # else:
                    #     hue_in = st.sidebar.selectbox('hue: ', df_KBS.columns.tolist())
                    fig_all=sns.displot(df_KBS, x=x_axis_choice, y=y_axis_choice, hue=hue_in, kind="kde",rug=True)
                    plt.ylabel(y_axis_choice, fontsize=14)
                    plt.xlabel(x_axis_choice, fontsize=14)
                    plt.tick_params(axis='both', labelsize=12)
                    plt.legend(loc=(1,.4),fontsize=14,frameon=False)
                    st.pyplot(fig_all)
                    #sns.lmplot(data=df_KBS, x=x_axis_choice, y= y_axis_choice,hue=hue_in,palette='Dark2',legend=False)

                elif sd =="lm Plot":
                    # x_axis_choice = st.sidebar.selectbox(
                    # "x axis",
                    # labels_mpg)
                    # y_axis_choice = st.sidebar.selectbox(
                    # "y axis",
                    # labels_mpg)
                    # if (flag==2):
                    #     hue_in = st.sidebar.selectbox('hue: ', ['Crop','Management','Stability','bins'])
                    # else:
                    #     hue_in = st.sidebar.selectbox('hue: ', df_KBS.columns.tolist())
                    #hue_in = st.sidebar.selectbox('hue: ', ['Crop','Management','Stability','bins'])
                    fig_all=sns.lmplot(data=df_KBS, x=x_axis_choice, y= y_axis_choice,hue=hue_in,palette='Dark2',legend=False)

                    # You can use matplotlib to adjust parts of the plot
                    plt.ylabel(y_axis_choice, fontsize=14)
                    plt.xlabel(x_axis_choice, fontsize=14)
                    plt.tick_params(axis='both', labelsize=12)
                    plt.legend(loc=(1,.4),fontsize=14,frameon=False)
                    #sns.jointplot(x = x_axis_choice, y = y_axis_choice, data = df_KBS)

                    #sns.displot(df_mpg, x="horsepower", y="mpg", kind="kde", rug=True)
                    st.pyplot(fig_all)
                elif sd =="Joint Plot":
                    # x_axis_choice = st.sidebar.selectbox(
                    # "x axis",
                    # labels_mpg)
                    # y_axis_choice = st.sidebar.selectbox(
                    # "y axis",
                    # labels_mpg)
                    #hue_in = st.sidebar.selectbox('hue: ', ['Crop','Management','Stability','bins'])
                    #hue_in = st.sidebar.selectbox('hue: ', df_KBS.columns)
                    fig_all=sns.jointplot(x = x_axis_choice, y = y_axis_choice, data = df_KBS, kind="reg")
                    

                    # You can use matplotlib to adjust parts of the plot
                    plt.ylabel(y_axis_choice, fontsize=14)
                    plt.xlabel(x_axis_choice, fontsize=14)
                    plt.tick_params(axis='both', labelsize=12)
                    plt.legend(loc=(1,.4),fontsize=14,frameon=False)
                    #sns.jointplot(x = x_axis_choice, y = y_axis_choice, data = df_KBS)

                    #sns.displot(df_mpg, x="horsepower", y="mpg", kind="kde", rug=True)
                    st.pyplot(fig_all)
                elif sd=="Line Plot":
                    # x_axis_choice = st.sidebar.selectbox(
                    # "x axis",
                    # df_KBS.columns)
                    # y_axis_choice = st.sidebar.selectbox(
                    # "y axis",
                    # labels_mpg,index=1)
                    #hue_in = st.sidebar.selectbox('hue: ', df_KBS.columns,index= len(df_KBS.columns.tolist())-1)
                    base = alt.Chart(df_KBS).properties(width=550)

                    line = base.mark_line().encode(
                        x= x_axis_choice  ,
                        y=y_axis_choice,
                        #color=hue_in
                    )
                    line
        elif (selection=='Run ML Algorithm'):
            target = st.selectbox("Please select a target variable for prediction",df_KBS.columns)
                #st.write(target)
                
                #options = st.multiselect( 'Columns to drop', df_KBS.columns.tolist())
            #size_test= st.sidebar.slider(label='Test size', key='Size',value=0.3, min_value=0.1, max_value=0.9, step=0.1)

            # if flag==2:
            #     if len(target)>0:
            #         important = st.multiselect(
            #         'Select features model building ',
            #         df_KBS.drop([target],axis=1).columns.tolist(),
            #         [ 'Ammonium',  'WFPS', 'Nitrate','NDVI'])
            # else:
            if len(target)>0:
                important = st.multiselect(
                'Select features model building ',
                df_KBS.drop([target],axis=1).columns.tolist())
            
            with st.form(key='MLmodel'):
                mlselect = st.sidebar.selectbox(
                            "Select a ML model", 
                            [ 
                                "Random Forest",
                                "XGBRegressor",
                                "SGDRegressor",
                                "KernelRidge",
                                "ElasticNet",
                                "BayesianRidge",
                                "GradientBoostingRegressor",
                                "SVR"
                            ]
                )
                
                Ch_para=st.sidebar.checkbox('Change ML Parameters')
                list_ml={}
                #target=st.text_input('Please provide name for a target variable','')
                #target = st.selectbox("Please select a target variable for prediction",df_KBS.columns)
                #st.write(target)
                
                #options = st.multiselect( 'Columns to drop', df_KBS.columns.tolist())
                size_test= st.sidebar.slider(label='Test size', key='Size',value=0.3, min_value=0.1, max_value=0.9, step=0.1)

                # if flag==2:
                #     if len(target)>0:
                #         important = st.multiselect(
                #         'Select features model building ',
                #         df_KBS.drop([target],axis=1).columns.tolist(),
                #         [ 'Ammonium',  'WFPS', 'Nitrate','NDVI'])
                # else:
                #     if len(target)>0:
                #         important = st.multiselect(
                #         'Select features model building ',
                #         df_KBS.drop([target],axis=1).columns.tolist())

                
                if (mlselect == "Random Forest") &(len(target)>0):
                    if Ch_para:
                        num_tree= st.sidebar.slider(label='Number of trees in random forest', key='tSize',value=1000, min_value=100, max_value=10000, step=100)
                        max_features= st.sidebar.slider(label='Number of features to consider at every split ', key='Mfeat',value=2, min_value=1, max_value=len(important), step=1)
                        max_depth= st.sidebar.slider(label='Maximum number of levels in tree ', key='Mdep',value=10, min_value=1, max_value=1000, step=10)
                        min_samples_split = st.sidebar.slider(label='Minimum number of samples required to split a node', key='Msplit',value=3, min_value=1, max_value=100, step=1)
                        min_samples_leaf= st.sidebar.slider(label='Minimum number of samples required at each leaf node ', key='Mleaf',value=2, min_value=1, max_value=100, step=1)
                        select_boot=st.sidebar.radio("Select bootstrap method for sampling",('Yes','No'))
                        if select_boot =='Yes':
                            bootstrap=True
                        else:
                            bootstrap=False  
                    else:
                        num_tree=1000
                        max_features= 2
                        max_depth=  2
                        min_samples_split = 20
                        min_samples_leaf=5
                        bootstrap=True
                    reg = RandomForestRegressor(n_estimators = num_tree, random_state = 42,\
                        max_features=max_features,max_depth= max_depth, min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf,\
                            bootstrap=bootstrap)
                
                elif(mlselect =="XGBRegressor"):
                    reg = XGBRegressor()
                elif(mlselect =="SGDRegressor"):
                    reg = SGDRegressor()
                elif(mlselect =="KernelRidge"):
                    reg = KernelRidge()
                elif(mlselect =="ElasticNet"):
                    reg = ElasticNet()
                elif(mlselect =="BayesianRidge"):
                    reg = BayesianRidge()
                elif(mlselect =="GradientBoostingRegressor",):
                    reg = GradientBoostingRegressor()
                elif(mlselect =="SVR"):
                    reg = SVR()

                # Split the data into training and testing 
                

                

            
        

                    
                    # Define the regression model that you want to use 
                #reg = RandomForestRegressor(n_estimators = num_tree, random_state = 42,max_features=4,bootstrap=True)
                Plot=st.checkbox('Plot')
                #Intialize the MachineLearningalgo class to perform final task 
                #MM=MachineLearningalgo(Xtrain, Xtest, Ytrain, Ytest,important,reg)
                submit_button = st.form_submit_button(label='Submit & Run')

            
            if submit_button:


                #st.write('### Selected columsn to drop:', options)
                st.write('### Selected features to run ML:', important)
                if (mlselect == "Random Forest"):
                    st.write('### Selected ML model details:',{"ML Model":mlselect,"n_estimators":num_tree, "max_features":max_features,\
                            "max_depth":max_depth, "min_samples_split":min_samples_split,"min_samples_leaf":min_samples_leaf,\
                                "bootstrap":bootstrap})
                else:
                    st.write('### Selected ML model:',{"ML Model":mlselect})
                                

                # exp_data().append({"ML Model":mlselect,"n_estimators":num_tree, "max_features":max_features,\
                #             "max_depth":max_depth, "min_samples_split":min_samples_split,"min_samples_leaf":min_samples_leaf,\
                #                 "bootstrap":bootstrap})

                #st.write('Selected  ML model and parameters:', exp_data())

                bins = [-5, 1, 5, 10, 15, 20, 35, 60, 600]

                labels = [
                        "verylow", "mediumlow", "low", "low-medium", "medium",
                        "medium-high", "high", "extreme"
                    ]
                # Initialize the DataProcess class to finalize the data to input into Machine learning algorithm
                dp=DataProcess(df_KBS)

                #Drop the features
                
                
            
                #if len(options)>0:
                #    df_filter=dp.drop_columns(options)
                #Use Label encoder to convert object or string to numeric values
                df_encode=dp.label_enc()
                    #visualizer = PredictionError(reg)
                
                #if flag==2:
                Xtrain, Xtest, Ytrain, Ytest = dp.split_data(target,testsize=size_test,strat=True,bins=bins,labels=labels)
            

                # Fit the training data to the visualizer
                reg.fit(Xtrain[important], Ytrain[target])
                
            
            # Use the forest's predict method on the test data
                
                predictions = reg.predict(Xtest[important])
                df_mpg=pd.DataFrame()
                df_mpg=Xtest[important]
                df_mpg['Predicted']=predictions
                df_mpg['Observed']=Ytest[target]
                
                #MM.visulaization_error()
                rmse_cal,r_cal,r_square_cal=correlation(df_mpg,'Observed','Predicted')
                st.write("### Results:")
                st.write("R${^{2}}$ = ",np.round(r_square_cal,2))
                st.write("R = ",np.round(r_cal[0],2))
                st.write("p = ",np.round(r_cal[1],5))
                st.write("RMSE = ",np.round(rmse_cal,2))
                #Add data
                if (mlselect == "Random Forest"):
                    exp_data().append({"Test_size":size_test,"ML Model":mlselect,"n_estimators":num_tree, "max_features":max_features,\
                            "max_depth":max_depth, "min_samples_split":min_samples_split,"min_samples_leaf":min_samples_leaf,\
                                "bootstrap":bootstrap,"Feature_list":important ,"R2":np.round(r_square_cal,2),"R":np.round(r_cal[0],2),"p":np.round(r_cal[1],5),\
                                    "RMSE":np.round(rmse_cal,2)})
                else:

                    exp_data().append({"Test_size":size_test,"ML Model":mlselect,"Feature_list":important ,"R2":np.round(r_square_cal,2),"R":np.round(r_cal[0],2),"p":np.round(r_cal[1],5),\
                                    "RMSE":np.round(rmse_cal,2)})
                if Plot:
                    if (mlselect == "Random Forest"):
                        col1,col2=st.columns([1,1])
                    # else:
                    #     col1=st.columns([1])

                        with col1:
                        
                            scatter = alt.Chart(df_mpg).properties(width=350).mark_circle(size=100).encode(x='Observed', y='Predicted',
                            tooltip=important).interactive()
                            reg_line=alt.Chart(df_mpg).properties(width=350).mark_circle(size=100).encode(x='Observed', y='Predicted',
                            tooltip=important).transform_regression('Observed','Predicted').mark_line()
                            scatter+reg_line


                        # all_feat_imp_df = pd.DataFrame(data=[tree.feature_importances_ for tree in 
                        #                          reg],
                        #                    columns=important)

                        # fig, ax = plt.subplots(figsize=(15, 5))
                        # (sns.boxplot(data=all_feat_imp_df,ax=ax)
                        # .set(title='Random Forest'+' '+'Feature Importance Distributions',
                        #      ylabel='Importance'))
                        # st.pyplot(fig)

                        #Create arrays from feature importance and feature names
                        
                        with col2:
                            feature_importance = np.array(reg.feature_importances_)
                            feature_names = np.array(important)

                            #Create a DataFrame using a Dictionary
                            data={'feature_names':feature_names,'feature_importance':feature_importance}
                            fi_df = pd.DataFrame(data)

                            #Sort the DataFrame in order decreasing feature importance
                            fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)

                            #Define size of bar plot
                            fig=plt.figure(figsize=(10,15))
                            #Plot Searborn bar chart
                            sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
                            #Add chart labels
                            plt.title('Random Forest'+' ' + 'FEATURE IMPORTANCE',fontsize=20)
                            plt.xlabel('')
                            plt.ylabel('')
                            plt.tick_params(axis='both', labelsize=20)

                            st.pyplot(fig)
                    else:
                        scatter = alt.Chart(df_mpg).properties(width=350).mark_circle(size=100).encode(x='Observed', y='Predicted',
                        tooltip=important).interactive()
                        reg_line=alt.Chart(df_mpg).properties(width=350).mark_circle(size=100).encode(x='Observed', y='Predicted',
                        tooltip=important).transform_regression('Observed','Predicted').mark_line()
                        scatter+reg_line
            
            
                        
            col1, col2, col3= st.columns([1,1,1])

            with col1:
                show_data=st.button('Show')
                
            with col2:
                remove_last=st.button('Delete last entry')

            with col3:
                reset_data=st.button('Reset')
                
        

            
            if show_data:
                st.write("## Results data")
                st.write(pd.DataFrame(exp_data()))
                
            if remove_last: 
                exp_data().pop() 
                st.write("## Results data")
                st.write(pd.DataFrame(exp_data()))
            if (reset_data):
                exp_data().clear() 
                st.write("## Results data")
                st.write(pd.DataFrame(exp_data()))
            min_rmse=st.checkbox('MinRMSE')
            if   min_rmse:
                df=pd.DataFrame(exp_data())
                st.write(df[df.RMSE == df.RMSE.min()])
                    
with tab4:
        embed_component= {'linkedin':"""<script src="https://platform.linkedin.com/badges/js/profile.js" async defer type="text/javascript"></script>
         <div class="badge-base LI-profile-badge" data-locale="en_US" data-size="medium" data-theme="light" data-type="VERTICAL" data-vanity="prateek-sharma-b581841b6" data-version="v1"><a class="badge-base__link LI-simple-link" href="https://www.linkedin.com/in/prateek-sharma-b581841b6?trk=profile-badge"</a></div>""", 'medium':"""<div style="overflow-y: scroll; height:500px;"> <div id="retainable-rss-embed"""}
        edu = [['M.S','Atmospheric Sciences','2017','UIUC','3.6 GPA'],['M.Tech','Atmospheric & Oceanic Sciences','2016','IIT Delhi', '8.9 CGPA'],['B.Tech','ME','2011','YMCA Faridabad','6.9 CGPA']]
        #<div class="badge-base LI-profile-badge" data-locale="en_US" data-size="medium" data-theme="light" data-type="VERTICAL" data-vanity="prateek-sharma-b581841b6" data-version="v1"><a class="badge-base__link LI-simple-link" href="https://www.linkedin.com/in/prateek-sharma-b581841b6?trk=profile-badge">Prateek Sharma</a></div>
              
        info = {'name':'Prateek Sharma', 
        'Brief':'Prateek Sharma is a PhD student at Michigan State University in Basso Lab. His reaserch interest lies in combining \
                process-based models with data science and machine learning tools to solve climate change problems ; \
                specifically related to  agricultural sector. Experienced in developing data-driven solutions/tools for quantifying GHGs in row-crop system', \
                'Mobile':'2176938571',
                'Email':'sharm165@msu.edu',
                'City':'East Lansing, Michigan',
                'edu':pd.DataFrame(edu,columns=['Qualification','Stream','Year','Institute','Score']),
                'skills':['Data Science','Pyhton','Fortran','ML','Streamlit'],
                'achievements':[],
                'publication_url':'https://medium.com/data-science-in-your-pocket/tagged/beginner'}

        skill_col_size=5
        st.write("""## Author: Prateek Sharma""")   
        col1, col2= st.columns([1,1])

        with col1:
            #st.write("Check out my linkedin profile: Click on View profile")
            st.image("./Profile_pic.png")
            link = '[Github repository](https://github.com/prat1987/CMSE830)'
            st.markdown(link, unsafe_allow_html=True)
            components.html(embed_component['linkedin'],height=310)
        with col2:
            st.write("***Summary***")
            st.write(info['Brief'])
            st.caption('Want to connect?')
            st.write('📧: sharm165@msu.edu')
            #components.html(embed_component['linkedin'],height=310)

        st.subheader('Skills & Tools ⚒️')
        def skill_tab():
            rows,cols = len(info['skills'])//skill_col_size,skill_col_size
            skills = iter(info['skills'])
            if len(info['skills'])%skill_col_size!=0:
                rows+=1
            for x in range(rows):
                columns = st.columns(skill_col_size)
                for index_ in range(skill_col_size):
                    try:
                        columns[index_].button(next(skills))
                    except:
                        break
        with st.spinner(text="Loading section..."):
            skill_tab()


        st.subheader('Education 📖')

        fig = go.Figure(data=[go.Table(
            header=dict(values=list(info['edu'].columns),
                        fill_color='paleturquoise',
                        align='left',height=65,font_size=20),
            cells=dict(values=info['edu'].transpose().values.tolist(),
                    fill_color='lavender',
                    align='left',height=40,font_size=15))])

        #fig.update_layout(width=750, height=2000)
        st.plotly_chart(fig)

        
