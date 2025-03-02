import boto3

# Initialize session with profile
session = boto3.Session(profile_name="myLearning", region_name="us-west-2")
ec2 = session.client("ec2")

# Retrieve all subnets
response = ec2.describe_subnets()

# Print subnet details
print("Listing all subnets:")
for subnet in response["Subnets"]:
    subnet_id = subnet["SubnetId"]
    vpc_id = subnet["VpcId"]
    cidr_block = subnet["CidrBlock"]
    availability_zone = subnet["AvailabilityZone"]
    
    print(f"Subnet ID: {subnet_id}, VPC ID: {vpc_id}, CIDR: {cidr_block}, AZ: {availability_zone}")
