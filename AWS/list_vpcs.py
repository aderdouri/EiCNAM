import boto3

# Create an EC2 client with region
ec2 = boto3.client("ec2", region_name="us-east-1")

# Retrieve all VPCs
response = ec2.describe_vpcs()

# Print VPC details
print("Listing all VPCs:")
for vpc in response["Vpcs"]:
    vpc_id = vpc["VpcId"]
    cidr_block = vpc["CidrBlock"]
    is_default = vpc.get("IsDefault", False)

    print(f"VPC ID: {vpc_id}, CIDR: {cidr_block}, Default: {is_default}")
