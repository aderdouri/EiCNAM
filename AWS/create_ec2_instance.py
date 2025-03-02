import boto3

# Initialize session with profile
session = boto3.Session(profile_name="myLearning", region_name="us-west-2")
ec2 = session.client("ec2")

# Find the Latest Amazon Linux AMI for g5 Instances
ami_response = ec2.describe_images(
    Owners=["amazon"],
    Filters=[
        {"Name": "name", "Values": ["amzn2-ami-hvm-*-x86_64-gp2"]},
        {"Name": "state", "Values": ["available"]},
    ],
)

ami_id = sorted(ami_response["Images"], key=lambda x: x["CreationDate"], reverse=True)[0]["ImageId"]
print(f"Latest Amazon Linux AMI ID: {ami_id}")

# Specify the subnet ID
subnet_id = "subnet-05fce10cb31190970"  # Replace with your subnet ID

# Retrieve the security group ID
sg_response = ec2.describe_security_groups(Filters=[{"Name": "group-name", "Values": ["myLearning-SG"]}])
sg_id = sg_response["SecurityGroups"][0]["GroupId"]
print(f"Using Security Group ID: {sg_id}")

# Create EC2 instance
instance_type = "g5.xlarge"

response = ec2.run_instances(
    ImageId=ami_id,
    InstanceType=instance_type,
    KeyName="myLearning-key-pair",
    MinCount=1,
    MaxCount=1,
    SubnetId=subnet_id,
    SecurityGroupIds=[sg_id],  # Use the security group ID
    TagSpecifications=[
        {
            "ResourceType": "instance",
            "Tags": [{"Key": "Name", "Value": "myLearning-Instance"}],
        }
    ],
)

instance_id = response["Instances"][0]["InstanceId"]
print(f"EC2 Instance launched! Instance ID: {instance_id}")
