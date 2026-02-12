from sagemaker.pytorch import PyTorchModel
import sagemaker
import boto3
from botocore.exceptions import ClientError

def deploy_endpoint():
    boto_session = boto3.Session(region_name="ap-south-1")
    sagemaker_session = sagemaker.Session(boto_session=boto_session)
    role = "arn:aws:iam::219967434603:role/sentiment-analyzer-endpoint"
    endpoint_name = "sentiment-analyzer-arpitv3-0"

    sm_client = boto_session.client("sagemaker")
    try:
        sm_client.describe_endpoint_config(EndpointConfigName=endpoint_name)
        print(f"Deleting existing endpoint configuration: {endpoint_name}")
        sm_client.delete_endpoint_config(EndpointConfigName=endpoint_name)
    except ClientError:
        pass

    try:
        sm_client.describe_endpoint(EndpointName=endpoint_name)
        print(f"Deleting existing endpoint: {endpoint_name}")
        sm_client.delete_endpoint(EndpointName=endpoint_name)
    except ClientError:
        pass

    model_uri = "s3://sentiment-analyzer-arpit/inference/model.tar.gz"

    model = PyTorchModel(
        model_data=model_uri,
        role=role,
        framework_version="2.5.1",
        py_version="py311",
        entry_point="inference.py",
        source_dir = ".",
        name = endpoint_name,
        sagemaker_session=sagemaker_session
    )

    predictor = model.deploy(
        initial_instance_count=1,
        instance_type="ml.m5.xlarge",
        endpoint_name=endpoint_name,
    )

if __name__ == "__main__":
    deploy_endpoint()