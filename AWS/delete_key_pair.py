import boto3

# Initialize EC2 client
ec2 = boto3.client('ec2')

# Specify the key pair name to delete
key_pair_name = "myBeBoulder2025Feb"  # Change this to your key pair name

# Delete the key pair
try:
    ec2.delete_key_pair(KeyName=key_pair_name)
    print(f"Key pair '{key_pair_name}' deleted successfully.")
except Exception as e:
    print(f"Error deleting key pair: {str(e)}")
