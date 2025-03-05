import boto3

# Initialize EC2 client with region
ec2 = boto3.client('ec2', region_name='us-east-1')

# Fetch all instances
response = ec2.describe_instances()

# Iterate through instances and print their status
print("EC2 Instances:")
for reservation in response['Reservations']:
    for instance in reservation['Instances']:
        instance_id = instance['InstanceId']
        state = instance['State']['Name']
        instance_type = instance['InstanceType']
        public_ip = instance.get('PublicIpAddress', 'N/A')
        print(f"- ID: {instance_id}, Type: {instance_type}, State: {state}, Public IP: {public_ip}")
