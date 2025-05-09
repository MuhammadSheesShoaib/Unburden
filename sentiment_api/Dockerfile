# Use Python 3.10 as the base image
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file first for efficient caching
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Copy the entire application code
COPY . .

# Expose the port used by FastAPI
EXPOSE 8000

# Command to run the FastAPI app using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
