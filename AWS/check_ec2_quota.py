import boto3

# Initialize session with profile
session = boto3.Session(profile_name="myLearning", region_name="us-east-1")
client = session.client("service-quotas")

response = client.get_service_quota(ServiceCode="ec2", QuotaCode="L-1216C47A")

print(f"Current vCPU Limit: {response['Quota']['Value']}")
