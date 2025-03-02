import boto3
import os

# Initialize session with profile
session = boto3.Session(profile_name="myLearning")
ec2 = session.client("ec2")

# Create a new key pair
key_name = "myLearning-key-pair"
response = ec2.create_key_pair(KeyName=key_name)

# Save the private key to a file in the ~/.ssh directory
ssh_dir = os.path.expanduser("~/.ssh")
os.makedirs(ssh_dir, exist_ok=True)
private_key_path = os.path.join(ssh_dir, f"{key_name}.pem")
private_key = response["KeyMaterial"]
with open(private_key_path, "w") as file:
    file.write(private_key)

print(f"Key pair '{key_name}' created and saved as {private_key_path}")
