FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    openssh-server \
    curl \
    unzip \
    vim \
    nano \
    git \
    jq \
    less \
    sudo \
    bsdutils \
    && rm -rf /var/lib/apt/lists/*

# AWS CLI v2 설치
RUN curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip \
    && unzip -q /tmp/awscliv2.zip -d /tmp/ \
    && /tmp/aws/install \
    && rm -rf /tmp/awscliv2.zip /tmp/aws

# devops-operator 계정 생성
RUN useradd -m -s /bin/bash -c "DevOps Operator" devops-operator

# SSH 서버 설정
RUN mkdir -p /var/run/sshd
COPY ssh-honeypot/sshd_config /etc/ssh/sshd_config
COPY ssh-honeypot/banner.txt  /etc/ssh/banner.txt
RUN ssh-keygen -A

# fake_filesystem 내용 복사 (먼저)
COPY --chown=devops-operator:devops-operator \
    moto/core/llm_agents/fake_filesystem/home/devops-operator/ \
    /home/devops-operator/

# .bashrc 덮어쓰기 (프롬프트 + 세션 로깅)
COPY --chown=devops-operator:devops-operator \
    ssh-honeypot/bashrc \
    /home/devops-operator/.bashrc

# 유출된 키의 공개키를 authorized_keys에 등록 (fake_filesystem 이후에 덮어쓰기)
RUN mkdir -p /home/devops-operator/.ssh && chmod 700 /home/devops-operator/.ssh
COPY ssh-honeypot/keys/honeypot_rsa.pub /home/devops-operator/.ssh/authorized_keys
RUN chmod 600 /home/devops-operator/.ssh/authorized_keys \
    && chown -R devops-operator:devops-operator /home/devops-operator/.ssh

# 호스트명 설정 (Nexora 프로파일과 일치)
RUN echo "nexora-prod-bastion" > /etc/hostname

# 세션 로그 디렉터리
RUN mkdir -p /var/log/honeypot && chmod 777 /var/log/honeypot

COPY ssh-honeypot/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 22
CMD ["/entrypoint.sh"]
