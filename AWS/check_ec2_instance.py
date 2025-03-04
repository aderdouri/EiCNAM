import boto3
import time

# Initialize session with profile
session = boto3.Session(profile_name="myLearning", region_name="us-east-1")
ec2 = session.client("ec2")

# Replace with your instance ID
instance_id = "your-instance-id"

while True:
    response = ec2.describe_instances(InstanceIds=[instance_id])
    state = response["Reservations"][0]["Instances"][0]["State"]["Name"]
    print(f"Instance state: {state}")
    
    if state == "running":
        public_ip = response["Reservations"][0]["Instances"][0].get("PublicIpAddress", "Not assigned yet")
        print(f"Instance is running! Public IP: {public_ip}")
        break
    
    time.sleep(10)  # Wait and retry
