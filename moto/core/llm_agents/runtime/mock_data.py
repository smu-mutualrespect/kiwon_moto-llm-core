from __future__ import annotations

import json
from typing import Any


def get_mock_template(category: str) -> dict[str, Any]:
    """Provide realistic templates for complex AWS data structures."""
    templates = {
        "iam_policy": {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": "s3:ListBucket",
                    "Resource": "arn:aws:s3:::example-bucket"
                }
            ]
        },
        "log_event": {
            "timestamp": 1715431200000,
            "message": "[INFO] Connection established from 192.168.1.100",
            "ingestionTime": 1715431201000
        },
        "resource_policy": {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "PublicRead",
                    "Effect": "Allow",
                    "Principal": "*",
                    "Action": "s3:GetObject",
                    "Resource": "arn:aws:s3:::example-bucket/*"
                }
            ]
        }
    }
    return templates.get(category, {"error": "template_not_found", "available": list(templates.keys())})
