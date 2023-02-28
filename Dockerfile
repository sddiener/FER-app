# Use the official Python image as the parent image
FROM python:3.10-slim-buster

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port that Streamlit will use
EXPOSE 8501

# Set the command to run when the container starts
CMD ["streamlit", "run", "--server.port", "8501", "app.py"]