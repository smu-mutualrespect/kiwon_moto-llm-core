terraform {
  backend "s3" {
    bucket         = "nexora-prod-terraform-state"
    key            = "eks/bastion/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "nexora-prod-tf-locks"
    encrypt        = true
  }

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.40"
    }
  }
}

provider "aws" {
  region = var.region
}

data "aws_eks_cluster" "prod" {
  name = "nexora-prod-eks"
}

resource "aws_security_group_rule" "bastion_to_eks_api" {
  type              = "egress"
  security_group_id = "sg-0b29d451e3f8a7c2d"
  protocol          = "tcp"
  from_port         = 443
  to_port           = 443
  cidr_blocks       = ["10.20.0.0/16"]
  description       = "bastion access to private EKS API and VPC endpoints"
}
