# Use a specific platform to ensure compatibility [cite: 57]
FROM --platform=linux/amd64 python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Create directories for input and output as specified [cite: 69]
RUN mkdir -p /app/input /app/output

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the pre-downloaded model files and the main script
COPY ./model/ ./model/
COPY run.py .

# Command to execute the script when the container runs
# This will process all PDFs in /app/input [cite: 69]
CMD ["python", "run.py"]