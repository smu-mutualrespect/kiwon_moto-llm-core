# PhantomGate 보고서 파이프라인

## 보고서에 들어가는 내용

보고서는 총 9개 섹션으로 구성됩니다.

| 섹션 | 내용 |
|------|------|
| 1. 개요 (Executive Summary) | 경영진/비기술자 대상 3문장 요약 — 언제, 누가, 무엇을, 어떤 영향 |
| 2. 사고 개요 | 사고 유형 / 공격 벡터 / 최초 탐지 시각 / 지속 시간 / 영향 서비스 / 위험도 |
| 3. 공격 흐름 분석 | Kill Chain 또는 ATT&CK 전술 순서로 단계별 서술, 증거 번호 인용 |
| 4. MITRE ATT&CK TTP 매핑 | 전술 / 기법 ID / 기법명 / Procedure / 증거 / 신뢰도 |
| 5. 침해지표 (IOC) | 자격증명 지표 / 행위 기반 지표 / 자산 기반 지표 |
| 6. 영향 평가 | 기밀성(Confidentiality) / 무결성(Integrity) / 가용성(Availability) |
| 7. 위험도 평가 | 심각 / 높음 / 중간 / 낮음 + 판단 근거 |
| 8. 대응 조치 권고 | 즉각(24h) / 단기(1주) / 중장기(1개월) |
| 9. 재발 방지 및 교훈 | Lessons Learned |

---

## 보고서 생성 시 남는 산출물

`generate_attack_report(session_id)` 호출 1회에 파일 3개가 생성됩니다.

```
reports/
├── {session_id[:16]}_{timestamp}.md
│       LLM이 생성한 한국어 Markdown IR 보고서
│
├── {session_id[:16]}_{timestamp}_navigator.json
│       MITRE ATT&CK Navigator 레이어
│       mitre-attack.github.io/attack-navigator 에서 시각화 가능
│       TTP별 score = 관찰 횟수 × 25 (최대 100)
│       0~24  → 노란색 #ffdd00
│       25~49 → 주황노란색 #ffaa00
│       50~74 → 주황색 #ff6600
│       75~   → 빨간색 #ff0000
│
└── {session_id[:16]}_{timestamp}.stix.json
        STIX 2.1 번들
        SIEM / OpenCTI / MISP에 직접 임포트 가능
```

파일명 예시:
```
AKIAIOSFODNN7ATT_20260523T071002Z.md
AKIAIOSFODNN7ATT_20260523T071002Z_navigator.json
AKIAIOSFODNN7ATT_20260523T071002Z.stix.json
```

---

## 프롬프트 구성 방식

보고서 생성 직전에 4개 함수가 순서대로 실행되어 프롬프트 재료를 만듭니다.

### 1단계 — 타이밍 분석 `_compute_timing_analysis()`

Action Log의 timestamp들로 요청 간격을 계산합니다.

- 평균 간격 / 최소·최대 / 표준편차 / 전체 지속 시간
- `is_automated` 판정 기준: 평균 간격 10초 이하 **또는** 샘플 5개 이상에서 표준편차 2초 미만

### 2단계 — IOC 추출 `_extract_iocs()`

| 추출 항목 | 출처 |
|-----------|------|
| access_key_id / credential_type | session_id (AKIA → 장기 키, ASIA → 임시 키) |
| target_account_id | world state의 `consistency_locks.account_id` |
| targeted_services | action_log 전체에서 service 필드 중복 제거 |
| high_risk_operations | action_log에서 risk_score ≥ 0.6인 항목 |
| discovered_arns | world state의 `exposed_assets` 중 arn:aws: 로 시작하는 것 최대 20개 |
| automation_indicator | is_automated=True 일 때만 포함 |
| attack_duration | 첫 번째 ~ 마지막 timestamp 차이 |

### 3단계 — TTP 매핑 `_map_ttps()`

`_OP_TO_TECHNIQUE` 정적 딕셔너리로 `(service, operation)` → ATT&CK 기법 ID를 매핑합니다.

```
(iam, GetUser)                    → (매핑 없음)
(secretsmanager, ListSecrets)     → T1526  Cloud Service Discovery
(guardduty, ListDetectors)        → T1580  Cloud Infrastructure Discovery
(ec2, DescribeInstances)          → T1580  Cloud Infrastructure Discovery
(ssm, StartSession/TerminateSession) → (매핑 없음 — HIGH_INTERACTION 별도 처리)
(kms, ListKeys)                   → (매핑 없음)
(dynamodb, ListTables)            → (매핑 없음)
```

매핑된 기법마다 `attack_db.get_technique(tech_id)`로 상세 메타데이터를 조회합니다.

신뢰도 판정:

| 기준 | 신뢰도 |
|------|--------|
| 동일 기법 2회 이상 직접 관찰 | High |
| 1회 관찰 | Medium |
| 간접 추론 | Low |

### 4단계 — 프롬프트 조립 `_build_report_prompt()`

아래 6개 블록을 하나의 문자열로 합칩니다.

```
[세션 정보]
  - 세션 ID, 가상 계정 ID, 세션 시작 시각
  - 총 API 호출 수, 탐색 서비스 목록
  - 공격 단계 흐름 (phase 순서)
  - 최종 위험 점수

[요청 타이밍 분석]
  - 평균/표준편차/전체 지속 시간/자동화 여부

[전체 공격 타임라인]
  - [01] ~ [N] 번호 포함
  - 각 항목: timestamp | [phase] service:operation | source | risk | 발견자산

[사전 TTP 매핑]
  - tech_id | tactic | name | 신뢰도 | 증거 | 탐지가이드

[침해지표 IOC]
  - key: value 형태로 전부 나열

[보고서 양식]
  - 9개 섹션 구조 강제 지정
  - 증거 번호([01], [02]...) 인용 지시
  - 한국어 작성 명시
```

이 프롬프트 전체를 LLM에 전송하면 완성된 Markdown 보고서가 반환됩니다.

---

## 환경 변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `MOTO_HONEYPOT_SESSION_TIMEOUT` | 300 | idle 몇 초 후 자동 보고서 생성 |
| `MOTO_HONEYPOT_REPORT_DIR` | reports/ | 산출물 저장 경로 (`markdown/`에는 사람이 보는 `.md`, `artifacts/`에는 metrics/Navigator/STIX JSON 저장) |
| `MOTO_LLM_REPORT_MODEL` | (기본 모델 사용) | 보고서 전용 모델 지정 |
| `MOTO_LLM_REPORT_MAX_TOKENS` | 5000 | 보고서 최대 토큰 수 |
