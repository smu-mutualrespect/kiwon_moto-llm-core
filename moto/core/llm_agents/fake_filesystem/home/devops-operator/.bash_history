aws sts get-caller-identity
aws eks list-clusters --region us-east-1
aws eks describe-cluster --name nexora-prod-eks --region us-east-1
aws eks update-kubeconfig --name nexora-prod-eks --region us-east-1
kubectl config current-context
kubectl get ns
kubectl get nodes
kubectl get nodes -o wide
kubectl get pods -A
kubectl get pods -n prod
kubectl describe deploy -n prod backend-api
aws ecr describe-repositories --region us-east-1
aws ecr describe-images --repository-name nexora-backend-api --region us-east-1
aws s3 ls
aws s3 ls s3://nexora-backup-prod/daily --region us-east-1
aws ec2 describe-instances --region us-east-1
aws lambda list-functions --region us-east-1
aws logs describe-log-groups --region us-east-1
docker compose ps
docker ps
cd terraform
terraform init
terraform plan
cd ..
terraform -chdir=terraform plan -var-file=prod.tfvars
vim deploy.sh
./monitor.sh
./scripts/check-prod-health.sh
./deploy.sh
./backup.sh
kubectl rollout restart deployment/backend-api -n prod
kubectl rollout status -n prod deployment/backend-api
kubectl logs -n prod deploy/backend-api --tail=80
systemctl status nginx
systemctl restart nginx
journalctl -u backend-api -n 80
journalctl -u backend-api -f
tail -f /var/log/nginx/access.log
tail -n 100 /var/log/auth.log
