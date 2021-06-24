FROM python:3.7-slim
RUN pip install streamlit
RUN pip install google-cloud-bigquery google-auth
COPY app.py ./app.py
CMD streamlit run app.py --server.port $PORT