# Use a base image of Python (Alpine version for a smaller size)
# Alpine is chosen for its lightweight nature, which reduces the overall image size.
FROM python:3.10-slim

# Set the working directory inside the container
# All subsequent commands and operations will be executed in this directory.
WORKDIR /flask_app_kafka_consumer

# Set environment variables for Flask
# FLASK_APP: Specifies the main application file to run when Flask starts.
# FLASK_RUN_HOST: Configures Flask to listen on all network interfaces, allowing external access.
# FLASK_RUN_PORT: Defines the port number on which Flask will listen for incoming requests.
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5000

# Set additional environment variables for Kafka connection
# KAFKA_BROKER: Address of the Kafka broker.
# TOPIC_NAME: The name of the Kafka topic that the consumer will listen to.
ENV KAFKA_BROKER="kafka:9092"
ENV TOPIC_NAME="train-sensor-data"
#ENV VEHICLE_NAME="e700_4801"

# Install system dependencies including Docker CLI
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    gnupg \
    lsb-release \
    && rm -rf /var/lib/apt/lists/*

# Install Docker CLI
RUN curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
RUN echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/debian $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
RUN apt-get update && apt-get install -y docker-ce-cli && rm -rf /var/lib/apt/lists/*

# Copy all project files into the working directory of the container
# This includes Python scripts, Flask configuration, and any additional resources needed by the application.
# Copy only the dashboard code into its workdir
COPY dashboard/ .
# Also add the OpenFAIR package to PYTHONPATH by copying the project root
COPY OpenFAIR/ /OpenFAIR/
# Copy the config directory for hydra
COPY config/ /config/
ENV PYTHONPATH=/

# Upgrade pip to the latest version
# Ensures that the latest packages and features are available.
RUN pip install --no-cache-dir --upgrade pip

# Install the dependencies specified in the requirements file
# The requirements file should list all Python packages needed for the Flask app and Kafka consumer.
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 5000 to allow external access to the Flask app
# This makes the port accessible outside the container.
EXPOSE 5000

# Command to start the application when the container runs
CMD ["python", "app.py"]
