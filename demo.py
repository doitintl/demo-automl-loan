import streamlit as st

st.title('Google AutoML Tables Loan Prediction')

income = st.number_input("Annual Income") 
income = st.number_input("Funded AMNT?")
income = st.number_input("Funded AMNT inv?")  

homeOwnership = st.selectbox('Home Ownership',("MORTAGE","RENT"))
grade = st.selectbox('Grade',("A","B","C","D","E","F","G"))

income = st.number_input("installment")
income = st.number_input("int rate") 
income = st.number_input("loan amount") 

purpose = st.selectbox('Purpose',("debt_consolidation","credit_card","home_improvement","major_purchase","small_business","medical","car","moving","vacation","house","wedding","renewable_energy","educational","other"))
paymentPlan = st.selectbox('Payment Plan',("Yes","NO"))
