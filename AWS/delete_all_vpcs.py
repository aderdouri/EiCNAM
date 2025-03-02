import boto3
import time

# Initialize session with profile
session = boto3.Session(profile_name="myLearning", region_name="us-west-2")
ec2 = session.client("ec2")

# Retrieve all VPCs
vpcs = ec2.describe_vpcs()["Vpcs"]

for vpc in vpcs:
    vpc_id = vpc["VpcId"]
    is_default = vpc.get("IsDefault", False)

    # Skip default VPC (cannot be deleted)
    if is_default:
        print(f"Skipping default VPC: {vpc_id}")
        continue

    print(f"Deleting VPC: {vpc_id}")

    # 1. Delete subnets
    subnets = ec2.describe_subnets(Filters=[{"Name": "vpc-id", "Values": [vpc_id]}])["Subnets"]
    for subnet in subnets:
        subnet_id = subnet["SubnetId"]
        try:
            ec2.delete_subnet(SubnetId=subnet_id)
            print(f"  Deleted Subnet: {subnet_id}")
        except boto3.exceptions.botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == 'DependencyViolation':
                print(f"  Subnet {subnet_id} has dependencies, retrying...")
                time.sleep(5)
                ec2.delete_subnet(SubnetId=subnet_id)
                print(f"  Deleted Subnet: {subnet_id}")
            else:
                raise

    # 2. Detach and delete internet gateways
    igws = ec2.describe_internet_gateways(Filters=[{"Name": "attachment.vpc-id", "Values": [vpc_id]}])["InternetGateways"]
    for igw in igws:
        igw_id = igw["InternetGatewayId"]
        ec2.detach_internet_gateway(InternetGatewayId=igw_id, VpcId=vpc_id)
        ec2.delete_internet_gateway(InternetGatewayId=igw_id)
        print(f"  Deleted Internet Gateway: {igw_id}")

    # 3. Delete route tables (except the main one)
    route_tables = ec2.describe_route_tables(Filters=[{"Name": "vpc-id", "Values": [vpc_id]}])["RouteTables"]
    for rt in route_tables:
        rt_id = rt["RouteTableId"]
        if any(assoc.get("Main", False) for assoc in rt.get("Associations", [])):
            continue  # Skip main route table
        ec2.delete_route_table(RouteTableId=rt_id)
        print(f"  Deleted Route Table: {rt_id}")

    # 4. Delete network ACLs (except the default)
    acls = ec2.describe_network_acls(Filters=[{"Name": "vpc-id", "Values": [vpc_id]}])["NetworkAcls"]
    for acl in acls:
        acl_id = acl["NetworkAclId"]
        if acl.get("IsDefault", False):
            continue  # Skip default ACL
        ec2.delete_network_acl(NetworkAclId=acl_id)
        print(f"  Deleted Network ACL: {acl_id}")

    # 5. Delete security groups (except default)
    security_groups = ec2.describe_security_groups(Filters=[{"Name": "vpc-id", "Values": [vpc_id]}])["SecurityGroups"]
    for sg in security_groups:
        sg_id = sg["GroupId"]
        if sg["GroupName"] == "default":
            continue  # Skip default SG
        ec2.delete_security_group(GroupId=sg_id)
        print(f"  Deleted Security Group: {sg_id}")

    # 6. Finally, delete the VPC
    ec2.delete_vpc(VpcId=vpc_id)
    print(f"✅ Deleted VPC: {vpc_id}")

print("🎉 All non-default VPCs deleted!")
