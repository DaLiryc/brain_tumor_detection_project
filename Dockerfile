FROM python:3.10.6-slim
# Using the official TensorFlow image as base


# First, pip install dependencies
COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt

# Only then copy brain!
COPY brain brain

# Run the API
CMD uvicorn brain.api.fast:app --host 0.0.0.0 --port $PORT
#--port $PORT
