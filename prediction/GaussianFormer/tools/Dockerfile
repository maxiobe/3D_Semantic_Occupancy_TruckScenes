# Use a lightweight official Python image as the base.
# Using python:3.8-slim matches your original environment's Python version.
FROM python:3.8-slim-bullseye

# Set environment variables for a clean Python environment
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies.
# 'build-essential' is required to compile packages like python-lzf.
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Install the required Python packages directly from PyPI in a single layer
RUN pip install --no-cache-dir tqdm truckscenes-devkit

# Copy your data generation script into the container.
# Make sure 'create_data.py' is in the same folder as this Dockerfile when you build.
COPY create_data.py .

# Set the main command (entrypoint) for the container.
# This makes the container behave like an executable for your script.
ENTRYPOINT ["python", "create_data.py"]

# Set the default arguments for the entrypoint.
# The container will show the help message if run without any arguments.
CMD ["--help"]

