import ast
import re
import sys
from datetime import datetime

import boto3

import src.services.architecture_validations as architecture_validations
from src.core.config import settings

_BUCKET_NAME = "pei-nwdaf-ml-architectures"


class ArchitectureService:
    def __init__(self):
        # setup minio connection
        self.s3 = boto3.client(
            "s3",
            endpoint_url=settings.MLFLOW_S3_ENDPOINT_URL,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_DEFAULT_REGION,
        )

        self._ensure_bucket()

    def load_architecture(self, architecture_id: str) -> tuple[type, bytes]:
        # get from minio and source
        pass

    def list_architectures(self):
        # list from minio. this doesnt change state
        pass

    def save_architecture(self, architecture_id: str, file: bytes, uploaded_by: str):
        """Validate and save the architecture to MinIO."""
        self._validate(architecture_id, file)

        try:
            self.s3.head_object(Bucket=_BUCKET_NAME, Key=architecture_id)
            raise ValueError(f"Architecture {architecture_id} already exists")
        except self.s3.exceptions.ClientError as e:
            if e.response["Error"]["Code"] != "404":
                raise

        self.s3.put_object(
            Bucket=_BUCKET_NAME,
            Key=architecture_id,
            Body=file,
            Metadata={
                "uploaded_by": uploaded_by,
                "uploaded_at": datetime.now().isoformat(),
            },
        )

    def delete_architecture(self, architecture_id: str):
        # delete from minio and source
        pass

    def _ensure_bucket(self):
        """Create the bucket if it does not exist."""
        try:
            self.s3.head_bucket(Bucket=_BUCKET_NAME)
        except self.s3.exceptions.ClientError as e:
            if e.response["Error"]["Code"] != "404":
                raise
            self.s3.create_bucket(Bucket=_BUCKET_NAME)

    def _validate(self, architecture_id: str, file: bytes) -> None:
        """Validate the file bytes."""

        ## validate size
        architecture_validations.size.run(file)

        ## parse file
        try:
            tree = ast.parse(file)
        except SyntaxError as e:
            raise ValueError(f"Syntax error in file: {e}")

        architecture_validations.content.run(tree)
        architecture_validations.interface.run(tree)
        architecture_validations.name.run(architecture_id)

    @staticmethod
    def help() -> str:
        return """
Architecture file constraints:
- Must contain exactly one class extending ModelInterface
- Class must implement train(self, X, y, max_epochs, status_callback)
-> float
- Class must implement predict(self, X)
- Do not import ModelInterface. It is injected at runtime
- Allowed imports: torch, numpy, logging, math, typing, abc ( other imports are forbidden)
- Forbidden calls: eval, exec, compile, __import__, open
- Forbidden dunder access: __class__, __bases__, __subclasses__, __globals__, __builtins__, __import__
- Max file size: 100KB
- Name: alphanumeric, underscore, hyphen only
"""
