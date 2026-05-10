#!/bin/bash
set -euo pipefail

source /home/moto/sangho-env/bin/activate
cd /home/moto/sangho_moto-llm-core

export $(cat .env | xargs)
export AWS_ACCESS_KEY_ID=testing
export AWS_SECRET_ACCESS_KEY=testing
export AWS_DEFAULT_REGION=us-east-1
export MOTO_LLM_AUDIT_FILE=/tmp/moto_audit_40.json
export AWS_ENDPOINT_URL=http://127.0.0.1:5000

rm -f /tmp/moto_audit_40.json /tmp/moto_server_40.log /tmp/cmd_results.tsv

# moto_server 시작
moto_server -p 5000 > /tmp/moto_server_40.log 2>&1 &
SERVER_PID=$!
echo "moto_server PID: $SERVER_PID"

# 서버 준비 대기
for i in $(seq 1 10); do
    if curl -sf http://127.0.0.1:5000/ > /dev/null 2>&1; then
        echo "서버 준비 완료"
        break
    fi
    sleep 1
done

AWS="aws"
RESULTS=()

run_cmd() {
    local idx=$1
    local label=$2
    shift 2
    local start end elapsed exit_code output
    start=$(date +%s%3N)
    output=$(timeout 60 $AWS "$@" 2>&1) && exit_code=0 || exit_code=$?
    end=$(date +%s%3N)
    elapsed=$((end - start))
    local status="OK"
    [[ $exit_code -ne 0 ]] && status="ERR"
    echo -e "[$idx] $label\t${elapsed}ms\t$status"
    echo -e "$idx\t$label\t$elapsed\t$status" >> /tmp/cmd_results.tsv
}

echo ""
echo "===== 명령어 실행 시작 ====="
echo ""

run_cmd  1  "bedrock:ListFoundationModels"                bedrock list-foundation-models
run_cmd  2  "ec2:MonitorInstances"                        ec2 monitor-instances --instance-ids i-1234567890abcdef0
run_cmd  3  "ec2:UnmonitorInstances"                      ec2 unmonitor-instances --instance-ids i-1234567890abcdef0
run_cmd  4  "ec2:DescribeReservedInstances"               ec2 describe-reserved-instances
run_cmd  5  "ec2:DescribeReservedInstancesListings"       ec2 describe-reserved-instances-listings
run_cmd  6  "ec2:PurchaseReservedInstancesOffering"       ec2 purchase-reserved-instances-offering --reserved-instances-offering-id aaaaaa11-bbbb-cccc-ddd-example1 --instance-count 1
run_cmd  7  "ec2:DescribeVolumeStatus"                    ec2 describe-volume-status --volume-ids vol-1234567890abcdef0
run_cmd  8  "ec2:ModifyVolumeAttribute"                   ec2 modify-volume-attribute --volume-id vol-1234567890abcdef0 --auto-enable-io
run_cmd  9  "ec2:CreateSpotDatafeedSubscription"          ec2 create-spot-datafeed-subscription --bucket my-honeypot-bucket
run_cmd 10  "ec2:DescribeBundleTasks"                     ec2 describe-bundle-tasks
run_cmd 11  "resource-explorer-2:ListIndexes"             resource-explorer-2 list-indexes
run_cmd 12  "resource-explorer-2:ListViews"               resource-explorer-2 list-views
run_cmd 13  "resource-explorer-2:Search"                  resource-explorer-2 search --query-string "*" --view-arn arn:aws:resource-explorer-2:us-east-1:123456789012:view/default/test-view
run_cmd 14  "support:DescribeServices"                    support describe-services --region us-east-1
run_cmd 15  "support:DescribeTrustedAdvisorCheckResult"   support describe-trusted-advisor-check-result --check-id Qch7DwouX1 --language en --region us-east-1
run_cmd 16  "support:DescribeTrustedAdvisorCheckSummaries" support describe-trusted-advisor-check-summaries --check-ids Qch7DwouX1 --region us-east-1
run_cmd 17  "eks:ListAddons"                              eks list-addons --cluster-name test-cluster
run_cmd 18  "eks:DescribeAddonVersions"                   eks describe-addon-versions
run_cmd 19  "ssm:StartSession"                            ssm start-session --target i-1234567890abcdef0
run_cmd 20  "ecs:ExecuteCommand"                          ecs execute-command --cluster test-cluster --task test-task --container test-container --interactive --command "/bin/sh"
run_cmd 21  "billingconductor:ListBillingGroups"          billingconductor list-billing-groups
run_cmd 22  "frauddetector:GetDetectors"                  frauddetector get-detectors
run_cmd 23  "detective:ListGraphs"                        detective list-graphs
run_cmd 24  "auditmanager:ListAssessments"                auditmanager list-assessments
run_cmd 25  "outposts:ListOutposts"                       outposts list-outposts
run_cmd 26  "appflow:ListFlows"                           appflow list-flows
run_cmd 27  "healthomics:ListRuns"                        healthomics list-runs
run_cmd 28  "mgn:DescribeSourceServers"                   mgn describe-source-servers
run_cmd 29  "codeguru-reviewer:ListRepositoryAssociations" codeguru-reviewer list-repository-associations
run_cmd 30  "backup-gateway:ListGateways"                 backup-gateway list-gateways
run_cmd 31  "ssm:DescribeInstanceInformation"             ssm describe-instance-information
run_cmd 32  "ecr:BatchCheckLayerAvailability"             ecr batch-check-layer-availability --repository-name demo --layer-digests sha256:abc
run_cmd 33  "ecr:GetDownloadUrlForLayer"                  ecr get-download-url-for-layer --repository-name demo --layer-digest sha256:abc
run_cmd 34  "ecr:InitiateLayerUpload"                     ecr initiate-layer-upload --repository-name demo
run_cmd 35  "ecr:CompleteLayerUpload"                     ecr complete-layer-upload --repository-name demo --upload-id test --layer-digests sha256:abc
run_cmd 36  "iam:GetContextKeysForPrincipalPolicy"        iam get-context-keys-for-principal-policy --policy-source-arn arn:aws:iam::123456789012:user/victim-admin
run_cmd 37  "iam:ListServiceSpecificCredentials"          iam list-service-specific-credentials --user-name victim-admin
run_cmd 38  "iam:GenerateServiceLastAccessedDetails"      iam generate-service-last-accessed-details --arn arn:aws:iam::123456789012:user/victim-admin
run_cmd 39  "secretsmanager:ValidateResourcePolicy"       secretsmanager validate-resource-policy --secret-id prod/db/password --resource-policy '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":"*","Action":"secretsmanager:GetSecretValue","Resource":"*"}]}'
run_cmd 40  "sts:DecodeAuthorizationMessage"              sts decode-authorization-message --encoded-message ZmFrZS1hdXRob3JpemF0aW9uLW1lc3NhZ2U=

# 서버 종료
kill $SERVER_PID 2>/dev/null || true

echo ""
echo "===== 완료 ====="
