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

import math
from sklearn.metrics import mean_squared_error
# get data
#df_KBS = sns.load_dataset("mpg")
#df_KBS = pd.read_csv("kbs_final_data.csv")
#df_KBS = pd.read_csv("kbs_final_data_salus_evi_soiltext.csv")
st.write("""
    # Tool for data analysis and run Machine learning models
    ### Dataset: Kellogs Biological Station (KBS) experimental nitrous oxide and other variables dataset 
    """)
st.markdown('**HELP: Please click on the following link for more information about the dataset used in this tool**')
link = '[MCSE KBS LTER experiment](https://lter.kbs.msu.edu/datatables)'
st.markdown(link, unsafe_allow_html=True)


st.write('N2O is one of the major greenhouse gases with a global warming potential (GWP) of 273 (± 118) times \
    more than carbon dioxide (CO2) for a 100-year time scale. This web app will let the user \
    explore the relationship between several variables measured and N2O emissions for a experiment using visualization and ML models which will help them to \
    understand the major factors impacting greenhouse gas (GHG) emissions from agricultural soils.'
)
st.write("""Author: Prateek Sharma""")


Exp_file_opt = st.radio(
                "Please select the option for the experiment file",
                ('Upload a file', 'Run the default'))
flag=0
#flag_default=1
if Exp_file_opt=='Upload a file':
    uploaded_file = st.file_uploader("Please provide an experiment file for analysis")
    flag=1
    file_format = st.radio(
                "Please select the format for the experiment file",
                ('CSV', 'Excel'))
#df_KBS = pd.read_csv("kbs_final_data_interpolate_evi.csv")
    if uploaded_file is not None:
        
        if (file_format=='CSV'):
            df_KBS = pd.read_csv(uploaded_file)
        elif(file_format=='Excel'):
            st.write(uploaded_file)
            df_KBS = pd.read_excel(open('Saha_et_al_2020_ERL_Data.xlsx','rb'),sheet_name='Data')
else:
    df_KBS = pd.read_csv("kbs_final_data_interpolate_evi.csv")
    flag=2
# else:
#     st.write(" ##Warning!!!Please select one of the two option for experiment files")
#     df_KBS=pd.DataFrame(data=None)
if ((flag==1 and  uploaded_file is not None) or flag==2):
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
        dnew=DataProcess(df_KBS)
        dnew.bin_prep(bins,labels)

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


        # allow user to choose which portion of the data to explore


        fig = plt.figure(figsize=(12, 6))

        if sd =='Hi Plot':
            # just convert it to a streamlit component with `.to_streamlit()` before
            if (flag==2):
                features_sel = st.sidebar.multiselect( 'Select features to plot parallel pairplot', df_KBS.columns.tolist(),
                ['N20','Ammonium',  'WFPS', 'Nitrate','NDVI'])
            else:
                features_sel = st.sidebar.multiselect( 'Select features to plot parallel plot', df_KBS.columns.tolist(),labels_mpg)
            xp=hip.Experiment.from_dataframe(df_KBS[features_sel])
            ret_val = xp.to_streamlit( key="hip").display()

            #st.markdown("hiplot returned " + json.dumps(ret_val))
            
        elif sd=='Pairplot':
            #fig_all = plt.figure()
            if (flag==2):
                features_sel = st.sidebar.multiselect( 'Select features to plot pairplot', df_KBS.columns.tolist(),
                ['Management','Ammonium',  'WFPS', 'Nitrate','NDVI'])
            else:
                features_sel = st.sidebar.multiselect( 'Select features to plot pairplot', df_KBS.columns.tolist(),
                labels_mpg)
            
            #labels_mpg.append('Management')
            hue_in = st.sidebar.selectbox('hue: ', features_sel,index=0)
            fig_all=sns.pairplot(df_KBS[features_sel], hue=hue_in) 
            st.pyplot(fig_all)

        elif sd == "Violin Plot":
            x_axis_choice = st.sidebar.selectbox(
            "x axis",
            labels_mpg)
            y_axis_choice = st.sidebar.selectbox(
            "y axis",
            df_KBS.columns,index= len(df_KBS.columns.tolist())-1)
            #hue_in = st.sidebar.selectbox('hue: ', df_KBS.columns)
          
            ax=sns.violinplot(x = x_axis_choice, y = y_axis_choice, data =df_KBS)

            plt.ylabel(y_axis_choice, fontsize=20)
            plt.xlabel(x_axis_choice, fontsize=20)
            plt.tick_params(axis='both', labelsize=18)
            plt.legend(loc=(1,.4),fontsize=14,frameon=False)
            st.pyplot(fig)
        elif sd == "box_plot":
            x_axis_choice = st.sidebar.selectbox(
            "x axis",
            labels_mpg)
            y_axis_choice = st.sidebar.selectbox(
            "y axis",
            df_KBS.columns,index= len(df_KBS.columns.tolist())-1)
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
            fig_all=sns.displot(df_KBS, x=x_axis_choice, y=y_axis_choice, hue=hue_in, kind="kde",rug=True)
            plt.ylabel(y_axis_choice, fontsize=14)
            plt.xlabel(x_axis_choice, fontsize=14)
            plt.tick_params(axis='both', labelsize=12)
            plt.legend(loc=(1,.4),fontsize=14,frameon=False)
            st.pyplot(fig_all)
            #sns.lmplot(data=df_KBS, x=x_axis_choice, y= y_axis_choice,hue=hue_in,palette='Dark2',legend=False)

        elif sd =="lm Plot":
            x_axis_choice = st.sidebar.selectbox(
            "x axis",
            labels_mpg)
            y_axis_choice = st.sidebar.selectbox(
            "y axis",
            labels_mpg)
            if (flag==2):
                hue_in = st.sidebar.selectbox('hue: ', ['Crop','Management','Stability','bins'])
            else:
                 hue_in = st.sidebar.selectbox('hue: ', df_KBS.columns.tolist())
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
            x_axis_choice = st.sidebar.selectbox(
            "x axis",
            labels_mpg)
            y_axis_choice = st.sidebar.selectbox(
            "y axis",
            labels_mpg)
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
            x_axis_choice = st.sidebar.selectbox(
            "x axis",
            df_KBS.columns)
            y_axis_choice = st.sidebar.selectbox(
            "y axis",
            labels_mpg,index=1)
            #hue_in = st.sidebar.selectbox('hue: ', df_KBS.columns,index= len(df_KBS.columns.tolist())-1)
            base = alt.Chart(df_KBS).properties(width=550)

            line = base.mark_line().encode(
                x= x_axis_choice  ,
                y=y_axis_choice,
                #color=hue_in
            )
            line
    elif (selection=='Run ML Algorithm'):
        
        with st.form(key='MLmodel'):
            target=st.text_input('Please provide name for a target variable','')
            st.write(target)
            
            options = st.multiselect( 'Columns to drop', df_KBS.columns.tolist())
            size_test= st.sidebar.slider(label='Test size', key='Size',value=0.3, min_value=0.1, max_value=0.9, step=0.1)

            if flag==2:
                if len(target)>0:
                    important = st.multiselect(
                    'Select features model building ',
                    df_KBS.drop([target],axis=1).columns.tolist(),
                    [ 'Ammonium',  'WFPS', 'Nitrate','NDVI'])
            else:
                if len(target)>0:
                    important = st.multiselect(
                    'Select features model building ',
                    df_KBS.drop([target],axis=1).columns.tolist(),
                    [ 'Ammonium',  'WFPS', 'Nitrate','NDVI'])

            mlselect = st.sidebar.selectbox(
                        "Select a ML model", 
                        [ 
                            "Random Forest"
                        ]
            )
            
            Ch_para=st.sidebar.checkbox('Change ML Parameters')
            list_ml={}
            if mlselect == "Random Forest":
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
            
            
            # Split the data into training and testing 
            

            

        
    

                
                # Define the regression model that you want to use 
            #reg = RandomForestRegressor(n_estimators = num_tree, random_state = 42,max_features=4,bootstrap=True)
            Plot=st.checkbox('Plot')
            #Intialize the MachineLearningalgo class to perform final task 
            #MM=MachineLearningalgo(Xtrain, Xtest, Ytrain, Ytest,important,reg)
            submit_button = st.form_submit_button(label='Submit & Run')

        
        if submit_button:


            st.write('### Selected columsn to drop:', options)
            st.write('### Selected features to run ML:', important)
            st.write('### Selected ML model details:',{"ML Model":mlselect,"n_estimators":num_tree, "max_features":max_features,\
                        "max_depth":max_depth, "min_samples_split":min_samples_split,"min_samples_leaf":min_samples_leaf,\
                            "bootstrap":bootstrap})
                            

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
            
            
        
            if len(options)>0:
                df_filter=dp.drop_columns(options)
            #Use Label encoder to convert object or string to numeric values
            df_encode=dp.label_enc()
                #visualizer = PredictionError(reg)
            

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
            exp_data().append({"Test_size":size_test,"ML Model":mlselect,"n_estimators":num_tree, "max_features":max_features,\
                    "max_depth":max_depth, "min_samples_split":min_samples_split,"min_samples_leaf":min_samples_leaf,\
                        "bootstrap":bootstrap,"Feature_list":important ,"R2":np.round(r_square_cal,2),"R":np.round(r_cal[0],2),"p":np.round(r_cal[1],5),\
                            "RMSE":np.round(rmse_cal,2)})
            
            if Plot:
                col1,col2=st.columns([1,1])

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
                   
                    