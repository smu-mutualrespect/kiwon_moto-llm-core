# PhantomGate Honeypot

**PhantomGate** is an AI Agent-based honeypot operation and vulnerability assessment framework. 

It is designed to deceive attackers into believing they have compromised a real AWS cloud server.  
When an attacker gains access via SSH, PhantomGate intelligently routes their commands based on complexity:

- **Low-Interactive Commands:** Handled seamlessly by Moto to simulate standard AWS responses.
- **High-Interactive / Unsupported Commands:** Handled dynamically by the AI Agent. If a command is unsupported by Moto or indicates a complex attack attempt, the Agent steps in to generate realistic fallback responses and lures.

Ultimately, PhantomGate safely analyzes the attacker's tactics, techniques, and procedures (TTPs) and provides comprehensive threat reports to the user.

---

## Honeypot Profile

The current AWS deception profile is a Nexora production DevOps bastion environment.
The canonical profile values live in `moto/core/llm_agents/honeypot_profile.py`, and
the AWS mock response helpers live in `moto/core/llm_agents/honeypot_aws_mocks.py`.

- OS: Ubuntu 22.04 LTS
- Hostname: `ip-10-20-4-37`
- User: `devops-operator`
- Region: `us-east-1`
- Account ID: `847362915408`
- IAM user: `devops-operator`
- Company prefix: `nexora`
- EKS cluster: `nexora-prod-eks`
- ECR registry: `847362915408.dkr.ecr.us-east-1.amazonaws.com`

---

## SSH Honeypot Entry Point (Docker)

PhantomGate provides a fully containerized SSH honeypot entry point.  
An attacker connects via SSH using a "leaked" private key, lands in a fake Nexora production environment, discovers AWS credentials in `~/.aws/credentials`, and begins probing the cloud infrastructure — all of which is silently routed to the moto honeypot server and recorded.

```
Attacker
  │  SSH (port 2222)  ← leaked private key
  ▼
SSH Container (Ubuntu 22.04 / devops-operator@nexora-prod-bastion)
  ├── ~/.aws/credentials   ← honeypot access key
  ├── ~/deploy.sh, ~/eks/, ~/terraform/, ...  ← fake Nexora filesystem
  └── aws s3 ls / aws iam list-users / ...
           │  HTTP (AWS_ENDPOINT_URL=http://moto:5000)
           ▼
     moto Honeypot Server
     + LLM Agent + Session Report
```

### Prerequisites

- Docker Engine installed and running

### Initial Setup (one-time)

Docker commands require membership in the `docker` group. Run the following steps once before the first build.

> **Note:** Replace `<your-username>` with the OS account you use to run the honeypot (e.g. `moto`).

```bash
# 1. Add your account to the docker group (requires sudo or root)
#    This grants permission to run docker commands without sudo.
sudo usermod -aG docker <your-username>

# 2. Apply the group change in the current terminal without re-logging in
#    (If you open a new terminal later, this step is not needed again.)
newgrp docker
```

### Build

All `docker compose` commands must be run from the **same directory that contains `docker-compose.yml`** — i.e., the root of this repository (wherever you cloned it).

```bash
# Move into the repository root first
cd <path-to-this-repo>

# Build both the moto server image and the SSH honeypot image
docker compose build
```

### Run & Stop

```bash
# Start both containers in the background
docker compose up -d

# Stop and remove all containers (data in reports/ and sessions/ is preserved)
docker compose down
```

### Test SSH Access

The simulated "leaked" private key is at `ssh-honeypot/keys/honeypot_rsa` inside the repository.  
All commands below must be run from the **repository root** (same directory as `docker-compose.yml`).

```bash
ssh -i ssh-honeypot/keys/honeypot_rsa -p 2222 devops-operator@localhost
```

Once connected, you land in the fake Nexora production environment as `devops-operator`. AWS CLI commands are silently routed to the moto honeypot — no real AWS account is contacted:

```bash
# These commands are answered by the moto honeypot, not real AWS
aws s3 ls
aws iam list-users
aws ec2 describe-instances
aws eks list-clusters
aws secretsmanager list-secrets
```

### Rebuild After Code Changes

When you modify honeypot source code, rebuild only the affected service and restart.  
Run from the repository root:

```bash
docker compose down && docker compose build ssh-honeypot && docker compose up -d
ssh -i ssh-honeypot/keys/honeypot_rsa -p 2222 devops-operator@localhost
```

> **Note:** The `moto` service uses an editable install, so Python source changes are picked up without rebuilding. Rebuild `moto` only when adding new dependencies.

### Monitoring & Logs

There are three places to look depending on what you want to know.

**1. SSH session recordings** — what the attacker typed, line by line

Every SSH session is recorded as a `.cast` file under `ssh-honeypot/sessions/`.  
The file is created the moment a user connects and grows in real time.

```bash
# List recorded sessions
ls ssh-honeypot/sessions/

# Read the raw text of a session (run from repository root)
cat ssh-honeypot/sessions/<session-filename>.cast
```

**2. moto honeypot server log** — AWS API calls, LLM agent responses, report events

This log shows every AWS CLI request the attacker made, which path handled it (moto native or LLM agent), and when a report was generated.  
Run from the repository root:

```bash
# Show all logs so far
docker compose logs moto

# Follow in real time (Ctrl+C to stop)
docker compose logs -f moto
```

**3. SSH container log** — connection and authentication events

Shows who connected, from which IP, and whether authentication succeeded or failed.  
Run from the repository root:

```bash
# Show all logs so far
docker compose logs ssh-honeypot

# Follow in real time (Ctrl+C to stop)
docker compose logs -f ssh-honeypot
```

**Attack reports** — full LLM-generated analysis of an attacker's session

Reports are generated automatically once a session has been idle for 300 seconds (configurable via `MOTO_HONEYPOT_SESSION_TIMEOUT`).  
Two types of output are saved:

```bash
# Markdown narrative report — attack flow, TTP mapping, IOCs, detection recommendations
ls reports/markdown/

# Structured artifacts — metrics JSON, STIX 2.1 bundle, ATT&CK Navigator layer
ls reports/artifacts/
```

---

## Running the Honeypot Server (Manual / Development)

First, switch to the user account you created and activate the virtual environment:

```bash
sudo su - <user account>
source <Name of the virtual environment you created>-env/bin/activate

```

Navigate to your working directory and update the package:

```bash
cd [moto+Agent_directory]

# Updates the package if there are any changes to the Agent code
pip install -e .

```

Load your environment variables. **This step is crucial for the fallback feature**:

```bash
# Registers the API keys managed in the .env file to the current terminal.
# This allows the system to fetch the keys when a Fallback occurs.
export $(cat .env | xargs)

```

> **Note:** This configuration is volatile and only valid for the currently open terminal session. If you close the window and open a new one, you must run this command again.

Finally, start the moto server:

```bash
# Start the server on port 5000. 
# (If a standard moto server is already running, you must change this port.)
moto_server -p 5000

```

---

## Honeypot Testing

To test the environment, open a **new terminal window** and activate the environment again:

```bash
sudo su - <user account>
source <Name of the virtual environment you created>-env/bin/activate

```

Set up your fake AWS credentials for testing:

```bash
# Configure fake credentials
export AWS_ACCESS_KEY_ID=testing
export AWS_SECRET_ACCESS_KEY=testing
export AWS_DEFAULT_REGION=us-east-1

```

Once this is set up, you can execute your desired AWS commands. Make sure you direct the commands to the honeypot server's endpoint port.

---

## TIP

You can create a terminal alias to avoid typing `--endpoint-url=http://127.0.0.1:5000` for every single command.

```bash
# 1. Create an alias named 'aws-local'
alias aws-local='aws --endpoint-url=http://127.0.0.1:5000'

# 2. After setting the alias, you can easily run commands like this:
aws-local s3 ls

```

---

## Supported Fallback Commands

Below is the list of AWS CLI commands that trigger the fallback mechanism. They are categorized by AWS service for easier reference.

### 1. Compute & Instances (EC2 & SSM)

```bash
# EC2 Instances & Volume Management
aws ec2 monitor-instances --instance-ids i-1234567890abcdef0
aws ec2 unmonitor-instances --instance-ids i-1234567890abcdef0
aws ec2 describe-reserved-instances
aws ec2 describe-reserved-instances-listings
aws ec2 purchase-reserved-instances-offering --reserved-instances-offering-id aaaaaa11-bbbb-cccc-ddd-example1 --instance-count 1
aws ec2 describe-volume-status --volume-ids vol-1234567890abcdef0
aws ec2 modify-volume-attribute --volume-id vol-1234567890abcdef0 --auto-enable-io
aws ec2 create-spot-datafeed-subscription --bucket honeypot-ki
aws ec2 describe-bundle-tasks

# Systems Manager (SSM)
aws ssm describe-instance-information

```

### 2. Containers (ECR)

```bash
aws ecr batch-check-layer-availability --repository-name demo --layer-digests sha256:abc
aws ecr get-download-url-for-layer --repository-name demo --layer-digest sha256:abc
aws ecr initiate-layer-upload --repository-name demo
aws ecr complete-layer-upload --repository-name demo --upload-id test --layer-digests sha256:abc

```

### 3. Identity, Security & Compliance (IAM, STS, Secrets Manager)

```bash
# IAM & STS
aws iam get-context-keys-for-principal-policy --policy-source-arn arn:aws:iam::123456789012:user/victim-admin
aws iam list-service-specific-credentials --user-name victim-admin
aws iam generate-service-last-accessed-details --arn arn:aws:iam::123456789012:user/victim-admin
aws sts decode-authorization-message --encoded-message ZmFrZS1hdXRob3JpemF0aW9uLW1lc3NhZ2U=

# Secrets Manager
aws secretsmanager validate-resource-policy --secret-id prod/db/password --resource-policy '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":"*","Action":"secretsmanager:GetSecretValue","Resource":"*"}]}'

# Access Analyzer, Detective, Audit Manager & Fraud Detector
aws accessanalyzer list-analyzers --region us-east-1
aws accessanalyzer list-findings --analyzer-arn <analyzer-arn> --region us-east-1
aws detective list-graphs
aws auditmanager list-assessments
aws frauddetector get-detectors

```

### 4. Resource Management & Governance

```bash
# Resource Explorer 2
aws resource-explorer-2 list-indexes
aws resource-explorer-2 list-views
aws resource-explorer-2 search --query-string "*" --view-arn <view-arn>

# CloudFormation & Organizations
aws cloudformation list-stacks
aws cloudformation describe-stack-resources --stack-name <stack-name>
aws organizations list-accounts
aws organizations list-roots

```

### 5. Backup & Data Management

```bash
aws backup list-backup-vaults --region us-east-1
aws backup-gateway list-gateways

```

### 6. Miscellaneous Services (AI/ML, Migration, Dev Tools)

```bash
# AI & Machine Learning
aws bedrock list-foundation-models
aws omics list-runs

# Developer Tools & Migration
aws codeguru-reviewer list-repository-associations
aws mgn describe-source-servers

# Other Integrations (Billing, AppFlow, Outposts)
aws billingconductor list-billing-groups
aws appflow list-flows
aws outposts list-outposts

```
