# Use Python 3.10 as the base image, slim for a smaller image size
FROM python:3.10-slim
 
# Set the working directory inside the container
WORKDIR /code
 
# Copy the requirements file
COPY ./requirements.txt /code/requirements.txt
 
# Install the Python dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
 
# Copy the modules folder into the container
COPY ./modules /code/modules
 
# Copy the model directory (with the saved model files) into the container
COPY ./models /code/models
 
# Expose port 5000 for Flask
EXPOSE 5000
 
# Command to run the Flask app with Uvicorn
CMD ["uvicorn", "modules.ml_pipeline_deployment:app", "--host", "0.0.0.0", "--port", "5000"]