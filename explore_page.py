import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def remove_outlier_hours_per_week(data):
    IQR = data['Hours per Week'].quantile(0.75) - data['Hours per Week'].quantile(0.25)
    lower_range = data['Hours per Week'].quantile(0.25) - (1.5 * IQR)
    upper_range = data['Hours per Week'].quantile(0.75) + (1.5 * IQR)
    data.loc[data['Hours per Week'] <= lower_range, 'Hours per Week'] = lower_range
    data.loc[data['Hours per Week'] >= upper_range, 'Hours per Week'] = upper_range

def shorten_categories(categories, cutoff):
    categorical_map = {}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            categorical_map[categories.index[i]] = categories.index[i]
        else:
            categorical_map[categories.index[i]] = 'Other'
    return categorical_map


def clean_education(x):
    if ' Bachelors' in x:
        return 'Bachelor’s degree'
    if ' Assoc-voc' in x or ' Assoc-acdm' in x:
        return 'Associates'
    if ' HS-grad' in x:
        return 'HS-Graduate'
    if ' Some-college' in x:
        return 'Colleges'
    if ' Masters'  in x:
        return 'Master’s'
    if ' Prof-school' in x:
        return 'Post grad'
    if ' Doctorate' in x:
        return 'Doctorate'
    return 'Dropout'



def clean_marital(x):
    if ' Married-civ-spouse' in x or ' Married-AF-spouse' in x :
        return 'Married'
    if ' Married-spouse-absent' in x or ' Separated' in x or ' Divorced' in x:
        return 'Not-married'
    if ' Widowed' in x:
        return ' Widowed'
    if ' Never-married' in x:
        return ' Never-married'


def clean_Occupation(x):
    if ' Adm-clerical' in x:
        return 'Admin'
    if ' Armed-Forces' in x:
        return 'Military'
    if ' Craft-repair' in x or ' Farming-fishing' in x or ' Handlers-cleaners' in x or ' Machine-op-inspct' in x or  ' Transport-moving' in x:
        return 'Blue-Collar'
    if ' Exec-managerial' in x:
        return 'White-Collar'
    if ' Other-service'  in x or ' Priv-house-serv'  in x:
        return 'Service'
    if ' Prof-specialty' in x:
        return 'Professional'
    if ' Protective-serv' in x or ' Tech-support' in x:
        return 'Other-Occupations'
    if ' Sales' in x:
        return 'Sales'


def clean_Employer(x):
    if ' Without-pay' in x:
        return ' Without-pay'
    if ' Self-emp-not-inc' in x or ' Self-emp-inc' in x:
        return 'Self-employed'
    if ' Local-gov' in x or ' State-gov' in x:
        return 'Other-gov'
    if ' Private' in x:
        return 'Private'
    if ' Federal-gov'  in x:
        return 'Federal-gov'


def remove_outlier_education_yrs(data):
    IQR = data['Education Yrs'].quantile(0.75) - data['Education Yrs'].quantile(0.25)
    lower_range = data['Education Yrs'].quantile(0.25) - (1.5 * IQR)
    upper_range = data['Education Yrs'].quantile(0.75) + (1.5 * IQR)
    data.loc[data['Education Yrs'] <= lower_range, 'Education Yrs'] = lower_range
    data.loc[data['Education Yrs'] >= upper_range, 'Education Yrs'] = upper_range


def remove_outlier_age(data):
    IQR = data['Age'].quantile(0.75) - data['Age'].quantile(0.25)
    lower_range = data['Age'].quantile(0.25) - (1.5 * IQR)
    upper_range = data['Age'].quantile(0.75) + (1.5 * IQR)
    data.loc[data['Age'] <= lower_range, 'Age'] = lower_range
    data.loc[data['Age'] >= upper_range, 'Age'] = upper_range


def remove_outlier_Capital_Gain(data):
    IQR = data['Capital Gain'].quantile(0.75) - data['Capital Gain'].quantile(0.25)
    lower_range = data['Capital Gain'].quantile(0.25) - (1.5 * IQR)
    upper_range = data['Capital Gain'].quantile(0.75) + (1.5 * IQR)
    data.loc[data['Capital Gain'] <= lower_range, 'Capital Gain'] = lower_range
    data.loc[data['Capital Gain'] >= upper_range, 'Capital Gain'] = upper_range

def remove_outlier_Capital_Loss(data):
    IQR = data['Capital Loss'].quantile(0.75) - data['Capital Loss'].quantile(0.25)
    lower_range = data['Capital Loss'].quantile(0.25) - (1.5 * IQR)
    upper_range = data['Capital Loss'].quantile(0.75) + (1.5 * IQR)
    data.loc[data['Capital Loss'] <= lower_range, 'Capital Loss'] = lower_range
    data.loc[data['Capital Loss'] >= upper_range, 'Capital Loss'] = upper_range

def remove_outlier_Salary(data):
    IQR = data['Salary'].quantile(0.75) - data['Salary'].quantile(0.25)
    lower_range = data['Salary'].quantile(0.25) - (1.5 * IQR)
    upper_range = data['Salary'].quantile(0.75) + (1.5 * IQR)
    data.loc[data['Salary'] <= lower_range, 'Salary'] = lower_range
    data.loc[data['Salary'] >= upper_range, 'Salary'] = upper_range

@st.cache
def load_data():
    train = pd.read_excel("Salary.xlsx")
    test = pd.read_excel("Hold_out.xlsx")
    train_len = len(train)
    test_len = len(test)
    data = pd.concat([train, test], axis=0)
    data.reset_index(drop=True, inplace=True)
    data = data.rename({"Native Country": "Country"}, axis=1)
    remove_outlier_hours_per_week(data)
    remove_outlier_education_yrs(data)
    remove_outlier_age(data)
    remove_outlier_Capital_Gain(data)
    remove_outlier_Capital_Loss(data)
    remove_outlier_Salary(data)
    data = data[['Education', 'Education Yrs', 'Marital Status', 'Employer',
                 'Occupation', 'Hours per Week', 'Country', 'Capital Gain', 'Capital Loss', 'Salary']]
    data = data[data["Salary"].notnull()]
    data = data.dropna()
    country_map = shorten_categories(data.Country.value_counts(), 100)
    data['Country'] = data['Country'].map(country_map)
    data['Country'] = np.where(data['Country'] == ' ?', 'Missing', data['Country'])
    indexCountry = data[data['Country'] == 'Missing'].index
    data.drop(indexCountry, inplace=True)
    data['Education'] = data['Education'].apply(clean_education)
    data['Marital Status'] = data['Marital Status'].apply(clean_marital)
    indexCountry = data[data['Occupation'] == ' ?'].index
    data.drop(indexCountry, inplace=True)
    data['Occupation'] = data['Occupation'].apply(clean_Occupation)
    data['Employer'] = data['Employer'].apply(clean_Employer)

    return data



def show_explore_page():
    st.title("Explore Employees Salaries")

    st.write(
        """ 
    ### Data is given by IIM kashipur
    """
    )
    data = load_data()

    data1 = data['Country'].value_counts()

    fig1, ax1 = plt.subplots()
    ax1.pie(data1, labels=data1.index, autopct="%1.1f%%", shadow=True, startangle=90)
    ax1.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.

    st.write("""#### Number of Data from different countries""")

    st.pyplot(fig1)

    st.write(
        """
    #### Mean Salary Based On Counrty
    """
    )

    data = data.groupby(["Country"])["Salary"].mean().sort_values(ascending=True)
    st.bar_chart(data)

