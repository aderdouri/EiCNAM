import boto3

# Initialize AWS session
session = boto3.Session(profile_name="myLearning", region_name="us-east-1")
ec2 = session.client("ec2")

# Define instance parameters
instance_type = "g5.xlarge"
key_pair_name = "myLearning-key-pair"
security_group_name = "myLearning-SG"
vpc_id = "vpc-00f301b5e0c9f5705"

try:
    # ✅ Find the Latest Amazon Linux AMI for g5 Instances
    ami_response = ec2.describe_images(
        Owners=["amazon"],
        Filters=[
            {"Name": "name", "Values": ["amzn2-ami-hvm-*-x86_64-gp2"]},
            {"Name": "state", "Values": ["available"]},
        ],
    )
    
    # ✅ Get the latest AMI ID
    ami_id = sorted(ami_response["Images"], key=lambda x: x["CreationDate"], reverse=True)[0]["ImageId"]
    print(f"Latest Amazon Linux AMI ID: {ami_id}")

    # ✅ Retrieve a subnet that allows public IPs
    subnet_response = ec2.describe_subnets(Filters=[{"Name": "vpc-id", "Values": [vpc_id]}])
    subnet_id = None

    for subnet in subnet_response["Subnets"]:
        if subnet["MapPublicIpOnLaunch"]:
            subnet_id = subnet["SubnetId"]
            print(f"Using Subnet with Public IP Enabled: {subnet_id}")
            break

    # ❌ If no public IP subnets are found, select the first available one (may be private)
    if not subnet_id:
        subnet_id = subnet_response["Subnets"][0]["SubnetId"]
        print(f"Using Private Subnet (No Public IP): {subnet_id}")

    # ✅ Retrieve the security group ID
    sg_response = ec2.describe_security_groups(Filters=[{"Name": "group-name", "Values": [security_group_name]}])
    sg_id = sg_response["SecurityGroups"][0]["GroupId"]
    print(f"Using Security Group ID: {sg_id}")

    # ✅ Launch EC2 instance (Fix: Remove instance-level `SecurityGroupIds`)
    response = ec2.run_instances(
        ImageId=ami_id,
        InstanceType=instance_type,
        KeyName=key_pair_name,
        MinCount=1,
        MaxCount=1,
        TagSpecifications=[
            {
                "ResourceType": "instance",
                "Tags": [{"Key": "Name", "Value": "myLearning-Instance"}],
            }
        ],
        NetworkInterfaces=[
            {
                "AssociatePublicIpAddress": True,  # ✅ Ensures a public IP is assigned
                "DeviceIndex": 0,
                "SubnetId": subnet_id,
                "Groups": [sg_id],  # ✅ Attach security group at the network interface level
            }
        ]
    )

    instance_id = response["Instances"][0]["InstanceId"]
    print(f"✅ EC2 Instance launched successfully! Instance ID: {instance_id}")

except boto3.exceptions.botocore.exceptions.ClientError as e:
    print(f"❌ Failed to launch EC2 instance: {e}")
except Exception as e:
    print(f"❌ An unexpected error occurred: {e}")
