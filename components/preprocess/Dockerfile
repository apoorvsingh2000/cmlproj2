FROM python:3.10-slim

# Install basic requirements
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy source code
COPY src/ /app/
WORKDIR /app

# Run the component
ENTRYPOINT ["python", "component.py"]
