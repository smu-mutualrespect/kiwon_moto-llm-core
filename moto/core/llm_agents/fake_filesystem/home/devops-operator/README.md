# Nexora Production Operations

Host: ip-10-20-4-37
User: devops-operator
Account: 847362915408
Region: us-east-1
Cluster: nexora-prod-eks
Registry: 847362915408.dkr.ecr.us-east-1.amazonaws.com
Backend image: 847362915408.dkr.ecr.us-east-1.amazonaws.com/nexora-backend-api:latest
Backup: s3://nexora-backup-prod/daily

## EKS

```bash
aws eks update-kubeconfig --name nexora-prod-eks --region us-east-1
kubectl get pods -A
kubectl get ns
kubectl get nodes
kubectl get pods -n prod
kubectl rollout restart deployment/backend-api -n prod
```

## Deploy

```bash
./deploy.sh
docker compose ps
docker ps
```

## Logs

```bash
journalctl -u backend-api -f
tail -f /var/log/nginx/access.log
tail -f /home/devops-operator/logs/deploy.log
tail -n 100 /var/log/auth.log
```

## Backup

```bash
./backup.sh
aws s3 ls s3://nexora-backup-prod/daily --region us-east-1
```

## Restart API

```bash
./restart-api.sh
systemctl restart nginx
systemctl restart backend-api
```

## AWS

```bash
aws sts get-caller-identity
aws eks list-clusters --region us-east-1
aws ecr describe-repositories --region us-east-1
aws s3 ls
aws ec2 describe-instances --region us-east-1
aws lambda list-functions --region us-east-1
aws logs describe-log-groups --region us-east-1
```

## Local

```bash
./monitor.sh
./cleanup-logs.sh
terraform plan
```
