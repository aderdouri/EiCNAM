import boto3

# Initialize session with profile
session = boto3.Session(profile_name="myLearning", region_name="us-east-1")
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
