import boto3

# Initialize EC2 client
ec2 = boto3.client('ec2')

# Fetch security groups
response = ec2.describe_security_groups()

# Print security groups details
print("EC2 Security Groups:")
for sg in response['SecurityGroups']:
    print(f"- Name: {sg['GroupName']}, ID: {sg['GroupId']}, VPC: {sg['VpcId']}")
    print("  Inbound Rules:")
    for rule in sg['IpPermissions']:
        protocol = rule.get('IpProtocol', 'N/A')
        from_port = rule.get('FromPort', 'N/A')
        to_port = rule.get('ToPort', 'N/A')
        ip_ranges = [ip['CidrIp'] for ip in rule.get('IpRanges', [])]
        print(f"    - Protocol: {protocol}, Ports: {from_port}-{to_port}, IPs: {ip_ranges}")

    print("  Outbound Rules:")
    for rule in sg['IpPermissionsEgress']:
        protocol = rule.get('IpProtocol', 'N/A')
        from_port = rule.get('FromPort', 'N/A')
        to_port = rule.get('ToPort', 'N/A')
        ip_ranges = [ip['CidrIp'] for ip in rule.get('IpRanges', [])]
        print(f"    - Protocol: {protocol}, Ports: {from_port}-{to_port}, IPs: {ip_ranges}")

    print("-" * 60)
