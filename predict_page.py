import streamlit as st
import pickle5 as pickle
import numpy as np


def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor = data["model"]
le_education = data["le_education"]
le_marital = data["le_marital"]
le_employer = data["le_employer"]
le_occupation = data["le_occupation"]
le_country = data["le_country"]

def show_predict_page():
    st.title("Employee Salary Prediction")

    st.write("""### We need some information to predict the salary""")



    education = (
        'Bachelor’s degree',
        'HS-Graduate',
        'Dropout',
        'Master’s',
        'Colleges',
        'Associates',
        'Doctorate',
        'Post grad',
    )

    married = (
        ' Never-married',
        'Married',
        'Not-married',
        ' Widowed',
    )

    employer = (
        'Self-employed',
        'Private',
        'Federal-gov',
        ' Without-pay',
    )

    occupation =(
        'Admin',
        'White-Collar',
        'Blue-Collar',
        'Professional',
        'Service',
        'Sales',
        'Military',
    )


    countries = (
        " United-States",
        " Mexico",
        " Philippines",
        " Germany",
        " Canada",
        " Puerto-Rico",
    )

    education = st.selectbox("Education Level", education)
    married = st.selectbox("About marriage", married)
    employer = st.selectbox("Employer", employer)
    occupation = st.selectbox("Occupation", occupation)
    hours_per_week = st. text_input("hours per week")
    country = st.selectbox("Country", countries)
    expericence = st.slider("Years of Experience", 0, 30, 3)

    ok = st.button("Calculate Salary")
    if ok:
        X = np.array([[education, expericence,married, employer,occupation,hours_per_week, country]])
        X[:, 0] = le_education.transform(X[:, 0])
        X[:, 2] = le_marital.transform(X[:, 2])
        X[:, 3] = le_employer.transform(X[:, 3])
        X[:, 4] = le_occupation.transform(X[:, 4])
        X[:, 6] = le_country.transform(X[:, 6])
        X = X.astype(float)

        salary = regressor.predict(X)
        st.subheader(f"The estimated salary is ${salary[0]:.2f}")
