import boto3

region = "us-east-1"  # Change this if needed
ec2 = boto3.client("ec2", region_name=region)

# List key pairs
key_pairs = ec2.describe_key_pairs()["KeyPairs"]
print("Existing Key Pairs:")
for key in key_pairs:
    print(f"- {key['KeyName']}")

# Delete all key pairs
for key in key_pairs:
    ec2.delete_key_pair(KeyName=key["KeyName"])
    print(f"Deleted Key Pair: {key['KeyName']}")