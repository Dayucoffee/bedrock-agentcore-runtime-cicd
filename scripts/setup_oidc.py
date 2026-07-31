#!/usr/bin/env python3
"""
GitHub OIDC Setup Script for AWS Bedrock AgentCore

This script configures OpenID Connect (OIDC) authentication between GitHub Actions
and AWS, enabling secure, keyless authentication for CI/CD pipelines.

Key Features:
- Creates GitHub OIDC identity provider in AWS
- Sets up IAM role with appropriate permissions for AgentCore deployments
- Configures trust relationships for specific GitHub repositories
- Follows AWS security best practices for CI/CD authentication

Usage:
    python setup_oidc.py --github-repo owner/repository-name

The script outputs an AWS Role ARN that should be added as a GitHub secret.
"""

import logging
from argparse import ArgumentParser
from json import dumps
from sys import exit

from boto3 import client as boto3_client
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def create_oidc_provider(iam_client):
    """
    Create or retrieve GitHub OIDC identity provider in AWS.

    This function sets up the OIDC identity provider that allows GitHub Actions
    to authenticate with AWS using temporary credentials instead of long-lived
    access keys.

    Args:
        iam_client: Boto3 IAM client instance

    Returns:
        str: ARN of the OIDC provider (existing or newly created)

    Raises:
        SystemExit: If provider creation fails
    """
    github_url = "https://token.actions.githubusercontent.com"
    thumbprint = [
        "6938fd4d98bab03faadb97b34396831e3780aea1",
        "1c58a3a8518e8759bf075b76b750d4f2df264fcd"
    ]

    try:
        providers = iam_client.list_open_id_connect_providers()
        for provider in providers["OpenIDConnectProviderList"]:
            if "token.actions.githubusercontent.com" in provider["Arn"]:
                logger.info(f"GitHub OIDC provider already exists: {provider['Arn']}")
                return provider["Arn"]

        logger.info("Creating GitHub OIDC identity provider...")
        response = iam_client.create_open_id_connect_provider(
            Url=github_url,
            ThumbprintList=thumbprint,
            ClientIDList=["sts.amazonaws.com"]
        )
        logger.info(f"Created OIDC provider: {response['OpenIDConnectProviderArn']}")
        return response["OpenIDConnectProviderArn"]

    except ClientError as e:
        if e.response["Error"]["Code"] == "EntityAlreadyExists":
            providers = iam_client.list_open_id_connect_providers()
            for provider in providers["OpenIDConnectProviderList"]:
                if "token.actions.githubusercontent.com" in provider["Arn"]:
                    logger.info(f"Using existing OIDC provider: {provider['Arn']}")
                    return provider["Arn"]
        logger.error(f"Error creating OIDC provider: {e}")
        exit(1)


def create_github_role(iam_client, provider_arn, github_repo):
    """
    Create or update IAM role for GitHub Actions with AgentCore permissions.

    This function creates an IAM role that GitHub Actions can assume using OIDC.
    If the role already exists, it updates the trust policy to match the current
    github_repo argument.

    Args:
        iam_client: Boto3 IAM client instance
        provider_arn (str): ARN of the GitHub OIDC provider
        github_repo (str): GitHub repository in format "owner/repo"

    Returns:
        str: ARN of the created or existing IAM role

    Raises:
        SystemExit: If role creation fails
    """
    role_name = "GitHubActions-AgentCore-Role"

    # Trust policy: defines who can assume this role
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Federated": provider_arn},
                "Action": "sts:AssumeRoleWithWebIdentity",
                "Condition": {
                    "StringEquals": {
                        "token.actions.githubusercontent.com:aud": "sts.amazonaws.com"
                    },
                    "StringLike": {
                        # Allow any branch/tag/workflow_dispatch from this repo
                        "token.actions.githubusercontent.com:sub": f"repo:{github_repo}:*"
                    }
                }
            }
        ]
    }

    # Permissions policy: defines what actions this role can perform
    permissions_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    # Bedrock and AgentCore permissions
                    "bedrock:InvokeModel",
                    "bedrock:InvokeModelWithResponseStream",
                    "bedrock:GetFoundationModel",
                    "bedrock:ListFoundationModels",
                    "bedrock-agentcore:*",
                    "bedrock-agentcore-control:*",
                    "bedrock-agentcore-control:CreateAgentRuntime",
                    "bedrock-agentcore-control:UpdateAgentRuntime",
                    "bedrock-agentcore-control:GetAgentRuntime",
                    "bedrock-agentcore-control:ListAgentRuntimes",
                    "bedrock-agentcore-control:DeleteAgentRuntime",
                    # ECR permissions for container management
                    "ecr:GetAuthorizationToken",
                    "ecr:BatchCheckLayerAvailability",
                    "ecr:GetDownloadUrlForLayer",
                    "ecr:BatchGetImage",
                    "ecr:PutImage",
                    "ecr:InitiateLayerUpload",
                    "ecr:UploadLayerPart",
                    "ecr:CompleteLayerUpload",
                    "ecr:CreateRepository",
                    "ecr:DescribeRepositories",
                    "ecr:DescribeImages",
                    "ecr:DeleteRepository",
                    "ecr:PutImageScanningConfiguration",
                    "ecr:PutRegistryScanningConfiguration",
                    # IAM permissions for role management
                    "iam:CreateRole",
                    "iam:GetRole",
                    "iam:PutRolePolicy",
                    "iam:PassRole",
                    # CloudWatch Logs for monitoring
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                    "logs:DescribeLogGroups",
                    "logs:DescribeLogStreams",
                    # Identity verification
                    "sts:GetCallerIdentity"
                ],
                "Resource": "*"
            }
        ]
    }

    try:
        response = iam_client.get_role(RoleName=role_name)
        role_arn = response["Role"]["Arn"]
        # Role exists — always update the trust policy to match the current repo
        logger.info(f"Role already exists: {role_arn}")
        logger.info(f"Updating trust policy for repo: {github_repo}")
        iam_client.update_assume_role_policy(
            RoleName=role_name,
            PolicyDocument=dumps(trust_policy)
        )
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchEntity":
            logger.info(f"Creating new role: {role_name}")
            response = iam_client.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=dumps(trust_policy),
                Description="GitHub Actions OIDC role for AgentCore deployments"
            )
            role_arn = response["Role"]["Arn"]
        else:
            logger.error(f"Error: {e}")
            exit(1)

    iam_client.put_role_policy(
        RoleName=role_name,
        PolicyName=f"{role_name}-Policy",
        PolicyDocument=dumps(permissions_policy)
    )

    logger.info("Trust policy and permissions updated successfully.")
    return role_arn


def main():
    parser = ArgumentParser()
    parser.add_argument("--github-repo", required=True, help="GitHub repo (owner/repo)")
    args = parser.parse_args()

    iam_client = boto3_client("iam")

    provider_arn = create_oidc_provider(iam_client)
    role_arn = create_github_role(iam_client, provider_arn, args.github_repo)

    logger.info("OIDC setup complete!")
    logger.info(f"Add this secret to GitHub: AWS_ROLE_ARN = {role_arn}")


if __name__ == "__main__":
    main()
