import streamlit as st
import streamlit.components.v1 as components
import requests
import json
import numpy as np

predictionEndpoint = 'https://demo-automl-loan-api-3dfvggur5q-ez.a.run.app/predict'


@st.cache(suppress_st_warning=True)
def getPrediction(
        annualIncome,
        fundetAmnt,
        fundetInvestor,
        grade,
        subGrade,
        homeOwnership,
        installment,
        intRate,
        loanAmount,
        purpose,
        term,
        verificationStatus,
        paymentPlan):
    data = {
        "instances": [
            {
                "annual_inc": annualIncome,
                "funded_amnt": fundetAmnt,
                "funded_amnt_inv": fundetInvestor,
                "grade": grade,
                "home_ownership": homeOwnership,
                "installment": installment,
                "int_rate": intRate,
                "loan_amnt": loanAmount,
                "purpose": purpose,
                "pymnt_plan": paymentPlan == 'Yes',
                "sub_grade": subGrade,
                "term": term,
                "verification_status": verificationStatus
            }
        ]
    }

    response = requests.post(predictionEndpoint, json=data)
    responseTime = response.elapsed.total_seconds()*1000
    statusCode = response.status_code

    return responseTime, statusCode, response


st.set_page_config(layout="wide")

st.title('Google Vertex AI Loan Prediction')
st.markdown('This Google Vertex AI example uses the Lending Club dataset https://www.kaggle.com/wordsforthewise/lending-club')
st.markdown('Lending Club is a peer-to-peer lending company that matches borrowers with investors through an online platform. It services people that need personal loans between $1,000 and $40,000. Borrowers receive the full amount of the issued loan minus the origination fee, which is paid to the company. Investors purchase notes backed by the personal loans and pay Lending Club a service fee. The company shares data about all loans issued through its platform during certain time periods.')

col1, col2 = st.columns(2)

with col1:
    annualIncome = st.number_input(
        label="The annual income provided by the borrower during registration", value=25411)
    fundetAmnt = st.number_input(
        "The total amount committed to that loan at that point in time", value=10000)
    fundetInvestor = st.number_input(
        "The total amount committed by investors for that loan at that point in time", value=10000)

    installment = st.number_input(
        "The monthly payment owed by the borrower if the loan originates", value=334)
    intRate = st.number_input("Interest Rate on the loan", value=12.39)
    loanAmount = st.number_input(
        "The listed amount of the loan applied for by the borrower", value=10000)


with col2:
    grade = st.selectbox('LC assigned loan grade',
                         ("A", "B", "C", "D", "E", "F", "G"))
    subGrade = st.selectbox('LC assigned loan subgrade', ("C1", "B4",
                            "B5", "B3", "C2", "C3", "C4", "B2", "B1", "C5", "A5", "D1"))

    homeOwnership = st.selectbox(
        'The home ownership status provided by the borrower during registration or obtained from the credit report.', ("MORTAGE", "RENT", "OWN", "OTHER"))

    purpose = st.selectbox('Purpose, a category provided by the borrower for the loan request.', ("debt_consolidation", "credit_card", "home_improvement",
                           "major_purchase", "small_business", "medical", "car", "moving", "vacation", "house", "wedding", "renewable_energy", "educational", "other"))
    paymentPlan = st.selectbox(
        'Indicates if a payment plan has been put in place for the loan', ("No", "Yes"))

    term = st.selectbox(
        'The number of payments on the loan. Values are in months and can be either 36 or 60.', ("36 months", "60 months"))
    verificationStatus = st.selectbox(
        'Verification Status, indicates if income was verified by LC', ("Source Verified", "Verified", "Not Verified"))

if st.button("Predict"):

    responseTime, statusCode, response = getPrediction(
        annualIncome,
        fundetAmnt,
        fundetInvestor,
        grade,
        subGrade,
        homeOwnership,
        installment,
        intRate,
        loanAmount,
        purpose,
        term,
        verificationStatus,
        paymentPlan)

    st.subheader('Prediction')


    prediction = json.loads(response.text)

    
    classIndex = np.argmax(prediction['predictions'][0]['scores'])
    classValue = prediction['predictions'][0]['classes'][classIndex]

    st.write(classValue)

    if int(classValue) == 0:
      st.subheader('Loan rejected :x:')

    if int(classValue) == 1:
      st.subheader('Loan accepted :white_check_mark:')

    
    st.write("Response time: ", responseTime, 'ms')
    

    st.write(prediction)


components.iframe("https://creditdecisioningdemo.cloud.looker.com/embed/dashboards-next/1", height=1000, scrolling=True)