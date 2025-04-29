import argparse
from google.cloud import aiplatform

def deploy_model(
    project: str,
    region: str,
    model_dir: str,
    model_display_name: str,
    endpoint_name: str,
    machine_type: str = "n1-standard-4"
):
    # Initialize the Vertex AI SDK
    aiplatform.init(project=project, location=region)

    # Upload the model to Vertex AI Model Registry
    model = aiplatform.Model.upload(
        display_name=model_display_name,
        artifact_uri=model_dir,
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-15:latest",  # TensorFlow 2.15 CPU serving image
    )

    model.wait()
    print(f"Model uploaded: {model.resource_name}")

    # Create or reuse an endpoint
    endpoint = aiplatform.Endpoint.create(display_name=endpoint_name)
    endpoint.wait()
    print(f"Endpoint created: {endpoint.resource_name}")

    # Deploy the model to the endpoint
    deployed_model = model.deploy(
        endpoint=endpoint,
        machine_type=machine_type,
        traffic_split={"0": 100}
    )
    print(f"Model deployed to endpoint: {endpoint.resource_name}")

    # Save endpoint info (optional)
    with open("/tmp/endpoint_info.txt", "w") as f:
        f.write(endpoint.resource_name)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, required=True)
    parser.add_argument('--project', type=str, required=True)
    parser.add_argument('--region', type=str, required=True)
    parser.add_argument('--endpoint-name', type=str, required=True)
    parser.add_argument('--model-display-name', type=str, required=True)
    parser.add_argument('--machine-type', type=str, default="n1-standard-4")
    args = parser.parse_args()

    deploy_model(
        project=args.project,
        region=args.region,
        model_dir=args.model_dir,
        model_display_name=args.model_display_name,
        endpoint_name=args.endpoint_name,
        machine_type=args.machine_type,
    )

if __name__ == "__main__":
    main()
