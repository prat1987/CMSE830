import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
class DataProcess():
    '''
    This class is used to do the final process on the data
    to feed it to the machine learning model.

    Attributes:
    
        df_in (dataframe): Input data set
    '''
    def __init__(self, df_in):
        '''
        The constructor for the DataProcess class.

        Parameters :
        
            df_in (dataframe): Input data set
        '''
        self.data = df_in
        #self.features = features_list
    def drop_columns(self, drop_list):
        '''
        The function is used to  drop the list of columns from a dataframe.

        Parameters:
        
            drop_list (list): List of columns to be dropped from the dataframe

       Returns:
       
            dataframe: A dataframe with removed columns
        '''
        self.data = self.data.drop(drop_list, axis=1)
        return self.data
    def label_enc(self):
        '''
        The function is used to  encode the string columns of the input dataframe 
        into numeric to prepare it to feed in ML model.

        Returns:
        
            dataframe: A dataframe with encoded columns
        '''
        le_code = LabelEncoder()
        for each in self.data.columns.values:
            encode = ['category', 'object', 'string']
            if self.data[each].dtypes in (encode):
                le_code.fit(self.data[each].astype(str))
                self.data[each] = le_code.transform(self.data[each].astype(str))
        return self.data
    def bin_prep(self, bins,labels,target):
        '''
        The function is used to  classify the target (N2O) values into different bins
        specified by the user.

        Parameters:
        
            bins (list) : A list of bins values
            
            labesl (list): A list of labes for the range of bins
        
        Returns:
        
            dataframe: A dataframe with bin classification column
        '''

      
        binned = pd.cut(self.data[target], bins=bins, labels=labels)
        # ADD this category in the dataframe
        self.data['bins'] = binned

    def split_data(self, target, testsize=0.2, strat=False, bins=[],labels=[]):
        '''
        This function is used to split the data set into training and testing datasets
        for machine learning algorithm. This stratify (start) parameter makes a split so that 
        the proportion of values in the sample produced will be the same as the proportion 
        of values provided to parameter stratify.
        
        For example, if our target variable N2O is a continous variable with values 0 and 10. 
        In first step we will define a bin for the whole N2O data, which based on the user expertise.
        Let say we have two bins range between 0-2 and 2-10 and there are  25% values lies in bin 1 and 
        75% in bin 2, stratify=bin will make sure that your random split has 25% of N2O value in range 0-2
        and 75% in range 2-10 in both training and testing. It depend on the user if they want 
        their data to be startified or not. 
        User can put True and  false the strat option based on their requirement. 

        Example for bins and labels 
        bins = [-5, 1, 5, 10, 15, 20, 35, 60, 600]

        labels = ["verylow", "mediumlow", "low", "low-medium", "medium",
                    "medium-high", "high", "extreme"]
        
        Parameters:
        
            target (string): Column name for the target variable in the dataframe
            
            testsize (float): Fraction of dataset that will be ditributed to test data. Value range from 0-1.
            
            start (bolean): Option to put on or off the stratification of data
            
            bins (list): User defined bins for the target variable.
            
            labels (list): User defined labels for the bins
       

        Returns:
        
            train_features (dataframe): A pandas dataframe of training data with features as columns 
            
            test_features (dataframe): A pandas dataframe of testing data with features as columns 
            
            train_labels (dataframe): A pandas dataframe of training data with target as a column 
            
            test_labels (dataframe): A pandas dataframe of testing data with target as a column 
        '''
        
        error=False
        
        if strat is True:
            # Stratify the data based on bin value
            if (len(labels)== (len(bins)-1))\
                &(len(bins)>0):
                self.bin_prep(bins,labels,target)
                ys_t = np.array(self.data['bins'])
                # axis 1 refers to the columns
                kbs_labels = pd.DataFrame(data=self.data[target], columns=[target])
                kbs_features = self.data.drop(target, axis=1)
                # Split the data into training and testing sets
                train_features, test_features, train_labels, test_labels = train_test_split(
                    kbs_features,
                    kbs_labels,
                    test_size=testsize,
                    random_state=42,
                    stratify=ys_t)
                # Checking data proportion in each bin in the original, training, and evaluation data
                print(self.data['bins'].value_counts(normalize=True), 'Original')
                print(train_features['bins'].value_counts(normalize=True),
                      'Training')
                print(test_features['bins'].value_counts(normalize=True),
                      'Testing')
            else:
                print('ERROR!!!: Bins and labels cannot be empty if stratify is true')
                error=True
                
        else:
            kbs_labels = pd.DataFrame(data=self.data[target], columns=[target])
            kbs_features = self.data.drop(target, axis=1)
            train_features, test_features, train_labels, test_labels = train_test_split(
                    kbs_features,
                    kbs_labels,
                    test_size=testsize,
                    random_state=42)
        if (error):
            print ('Please fix the error: No data is generated')
            train_features=pd.DataFrame(data=None)
            test_features=pd.DataFrame(data=None)
            train_labels=pd.DataFrame(data=None)
            test_labels=pd.DataFrame(data=None)
            
        return train_features, test_features,train_labels, test_labels
