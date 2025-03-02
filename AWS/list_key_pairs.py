import boto3

# Initialize EC2 client
ec2 = boto3.client('ec2')

# Fetch key pairs
response = ec2.describe_key_pairs()

# Print key pairs
print("EC2 Key Pairs:")
for key in response['KeyPairs']:
    print(f"- Name: {key['KeyName']}, Fingerprint: {key['KeyFingerprint']}")
