import boto3

# Initialize session with profile and region
session = boto3.Session(profile_name="myLearning", region_name="us-east-1")
ec2 = session.client("ec2")

# Fetch all running instances
response = ec2.describe_instances(Filters=[{"Name": "instance-state-name", "Values": ["running"]}])

# Collect instance IDs
instance_ids = [instance["InstanceId"] for reservation in response["Reservations"] for instance in reservation["Instances"]]

if instance_ids:
    # Stop all running instances
    ec2.stop_instances(InstanceIds=instance_ids)
    print(f"Stopping instances: {', '.join(instance_ids)}")
else:
    print("No running instances found.")
