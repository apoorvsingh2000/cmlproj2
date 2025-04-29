import argparse
from google.cloud import aiplatform_v1beta1 as aiplatform
import json

def predict_text(endpoint_id: str, project: str, region: str, instances: list):
    client = aiplatform.PredictionServiceClient()
    endpoint = client.endpoint_path(project=project, location=region, endpoint=endpoint_id)

    # Construct the request
    response = client.predict(
        endpoint=endpoint,
        instances=instances,
        parameters={}
    )

    print("Response from model prediction:")
    print(response)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--endpoint-id', type=str, required=True)
    parser.add_argument('--project', type=str, required=True)
    parser.add_argument('--region', type=str, required=True)
    parser.add_argument('--input-text', type=str, required=True)
    args = parser.parse_args()

    # This is a dummy tokenizer input; normally you'd reuse your real preprocessor
    # Here we assume model expects tokenized integer IDs (e.g., [14, 7, 123])
    input_instance = {
        "input": [1, 5, 8, 19, 42, 0, 0, 0]  # padded input example
    }

    print(f"Sending input to endpoint: {args.endpoint_id}")
    predict_text(
        endpoint_id=args.endpoint_id,
        project=args.project,
        region=args.region,
        instances=[input_instance]
    )

if __name__ == '__main__':
    main()
