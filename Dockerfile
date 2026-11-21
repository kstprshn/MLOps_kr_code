FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir scikit-learn==1.9.1 pandas==3.0.6 numpy==2.5.4 joblib==1.6.0 fastapi uvicorn
COPY app.py model.joblib features.joblib ./
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
