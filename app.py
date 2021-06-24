import streamlit as st
import requests
import json
import numpy as np
import logging
import google.auth
import google.auth.transport.requests
from google.cloud import bigquery

logging.getLogger().setLevel(logging.INFO)
predictionEndpoint = 'https://us-central1-aiplatform.googleapis.com/v1beta1/projects/doitintl-demo/locations/us-central1/endpoints/6952766176288047104:explain'


def get_token():
    creds, project = google.auth.default()
    auth_req = google.auth.transport.requests.Request()
    creds.refresh(auth_req)
    return creds.token

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
        paymentPlan,
        emp_title,
        emp_lenght,
        title
):
    data = {
        "instances": [
            {
                "annual_inc":  str(annualIncome),
                "funded_amnt":  str(fundetAmnt),
                "funded_amnt_inv":  str(fundetInvestor),
                "grade":  str(grade),
                "home_ownership":  str(homeOwnership),
                "installment":  str(installment),
                "int_rate":  str(intRate),
                "loan_amnt":  str(loanAmount),
                "purpose":  str(purpose),
                "pymnt_plan":  str(paymentPlan == 'Yes'),
                "sub_grade":  str(subGrade),
                "term":  str(term),
                "verification_status":  str(verificationStatus),
                "emp_length": str(emp_lenght),
                "emp_title":  str(emp_title),
                "title": str(title)
            }
        ]
    }
    token = get_token()
    response = requests.post(predictionEndpoint, json=data, headers={
        "Authorization": "Bearer "+ token
    })
    responseTime = response.elapsed.total_seconds()*1000
    statusCode = response.status_code

    return responseTime, statusCode, response


st.set_page_config(layout="wide")

st.title('Google AutoML Tables Loan Prediction')
st.markdown('This Google AutoML Tables example uses the Leanding Club dataset https://www.kaggle.com/wordsforthewise/lending-club')
st.markdown('Lending Club is a peer-to-peer lending company that matches borrowers with investors through an online platform. It services people that need personal loans between $1,000 and $40,000. Borrowers receive the full amount of the issued loan minus the origination fee, which is paid to the company. Investors purchase notes backed by the personal loans and pay Lending Club a service fee. The company shares data about all loans issued through its platform during certain time periods.')

col1, col2 = st.beta_columns(2)

with col1:
    annualIncome = st.number_input(
        label="The annual income provided by the borrower during registration", value=60000)
    fundetAmnt = st.number_input(
        "The total amount committed to that loan at that point in time", value=12000)
    fundetInvestor = st.number_input(
        "The total amount committed by investors for that loan at that point in time", value=0)

    installment = st.number_input(
        "The monthly payment owed by the borrower if the loan originates", value=347)
    intRate = st.number_input("Interest Rate on the loan", value=12.61)
    loanAmount = st.number_input(
        "The listed amount of the loan applied for by the borrower", value=12000)


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

    emp_lenght = st.selectbox(
        'Employment lenght', ("10+ Years", "1 < Year", "3 Years", "5 Years" ))

    emp_title =  st.text_input("Employment title", "Manager")
    title = st.text_input("Title", "Debt")


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
        paymentPlan,
        emp_title,
        emp_lenght,
        title
    )

    st.subheader('Prediction')


    prediction = json.loads(response.text)
    logging.info(prediction)
    

    classIndex = np.argmax(prediction['predictions'][0]['scores'])
    classValue = prediction['predictions'][0]['classes'][classIndex]

    st.write(classValue)

    if int(classValue) == 0:
      st.subheader('Loan rejected :x:')

    if int(classValue) == 1:
      st.subheader('Loan accepted :white_check_mark:')

    
    st.write("Response time: ", responseTime, 'ms')
    

    st.write(prediction)
    bq_row_to_insert = {
        "input" : {
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
            "verification_status": verificationStatus,
            "emp_length": emp_lenght,
            "emp_title": emp_title,
            "title": title
        },
        "prediction" : prediction['predictions'][0]['scores'][0],
        "attributions": prediction['explanations'][0]['attributions'][0]['featureAttributions']
    }
    logging.info(bq_row_to_insert)

    bq_client = bigquery.Client()
    table = bq_client.get_table("{}.{}.{}".format("doitintl-demo", "loan_demo_looker", "loan_predictions"))
    errors = bq_client.insert_rows_json(table, [bq_row_to_insert])
    logging.error(errors)