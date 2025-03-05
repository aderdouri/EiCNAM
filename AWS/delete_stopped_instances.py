import boto3

# Initialize session with profile and region
session = boto3.Session(profile_name="myLearning", region_name="us-east-1")
ec2 = session.client("ec2")

# Fetch all stopped instances
response = ec2.describe_instances(Filters=[{"Name": "instance-state-name", "Values": ["stopped"]}])

# Collect instance IDs
instance_ids = [instance["InstanceId"] for reservation in response["Reservations"] for instance in reservation["Instances"]]

if instance_ids:
    # Terminate all stopped instances
    ec2.terminate_instances(InstanceIds=instance_ids)
    print(f"Terminating instances: {', '.join(instance_ids)}")
else:
    print("No stopped instances found.")
