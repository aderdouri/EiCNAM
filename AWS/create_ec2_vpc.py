import boto3

# Initialize session with profile
session = boto3.Session(profile_name="myLearning", region_name="us-east-1")
ec2 = session.client("ec2")

# Specify the availability zone
availability_zone = "us-east-1a"

# Create VPC
vpc_response = ec2.create_vpc(CidrBlock="10.0.0.0/16")
vpc_id = vpc_response["Vpc"]["VpcId"]
ec2.create_tags(Resources=[vpc_id], Tags=[{"Key": "Name", "Value": "myLearning-VPC"}])
print(f"Created VPC: {vpc_id}")

# Create public subnet
public_subnet_response = ec2.create_subnet(
    CidrBlock="10.0.1.0/24", VpcId=vpc_id, AvailabilityZone=availability_zone
)
public_subnet_id = public_subnet_response["Subnet"]["SubnetId"]
ec2.create_tags(Resources=[public_subnet_id], Tags=[{"Key": "Name", "Value": "myLearning-Public-Subnet"}])
print(f"Created Public Subnet: {public_subnet_id}")

# Create private subnet
private_subnet_response = ec2.create_subnet(
    CidrBlock="10.0.2.0/24", VpcId=vpc_id, AvailabilityZone=availability_zone
)
private_subnet_id = private_subnet_response["Subnet"]["SubnetId"]
ec2.create_tags(Resources=[private_subnet_id], Tags=[{"Key": "Name", "Value": "myLearning-Private-Subnet"}])
print(f"Created Private Subnet: {private_subnet_id}")

# Create internet gateway
igw_response = ec2.create_internet_gateway()
igw_id = igw_response["InternetGateway"]["InternetGatewayId"]
ec2.attach_internet_gateway(InternetGatewayId=igw_id, VpcId=vpc_id)
ec2.create_tags(Resources=[igw_id], Tags=[{"Key": "Name", "Value": "myLearning-IGW"}])
print(f"Created and attached Internet Gateway: {igw_id}")

# Create route table and route for public subnet
route_table_response = ec2.create_route_table(VpcId=vpc_id)
route_table_id = route_table_response["RouteTable"]["RouteTableId"]
ec2.create_route(RouteTableId=route_table_id, DestinationCidrBlock="0.0.0.0/0", GatewayId=igw_id)
ec2.associate_route_table(RouteTableId=route_table_id, SubnetId=public_subnet_id)
ec2.create_tags(Resources=[route_table_id], Tags=[{"Key": "Name", "Value": "myLearning-Route-Table"}])
print(f"Created Route Table and associated with Public Subnet: {route_table_id}")

# Create security group
sg_response = ec2.create_security_group(
    GroupName="myLearning-SG", Description="Security group for SSH access", VpcId=vpc_id
)
sg_id = sg_response["GroupId"]
ec2.authorize_security_group_ingress(
    GroupId=sg_id,
    IpPermissions=[
        {
            "IpProtocol": "tcp",
            "FromPort": 22,
            "ToPort": 22,
            "IpRanges": [{"CidrIp": "0.0.0.0/0"}],
        }
    ],
)
ec2.create_tags(Resources=[sg_id], Tags=[{"Key": "Name", "Value": "myLearning-SG"}])
print(f"Created Security Group: {sg_id}")
