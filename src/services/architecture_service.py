import ast
import sys
import types
from contextlib import contextmanager
from datetime import datetime

import boto3
from sqlalchemy.orm import Session

import src.services.architecture_validations as architecture_validations
from src.core.config import settings
from src.models import ModelInterface

import logging

_BUCKET_NAME = "pei-nwdaf-ml-architectures"
logger = logging.getLogger(__name__)


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

    @contextmanager
    def load_architecture(self, architecture_id: str):
        cls, content = self._load(architecture_id)
        try:
            yield cls, content
        finally:
            self._unload(architecture_id)

    def _load(self, architecture_id: str) -> tuple[type, bytes]:
        response = self.s3.get_object(Bucket=_BUCKET_NAME, Key=architecture_id)
        content = response["Body"].read()

        module_name = self.module_name(architecture_id)
        namespace = self._make_namespace(module_name)

        exec(content, namespace)

        mod = types.ModuleType(module_name)
        mod.__dict__.update(namespace)
        sys.modules[module_name] = mod

        cls = next(
            (
                v
                for v in namespace.values()
                if isinstance(v, type)
                and issubclass(v, ModelInterface)
                and v is not ModelInterface
            ),
            None,
        )
        if cls is None:
            raise ValueError(f"No ModelInterface subclass found in {architecture_id}")

        return cls, content

    def _unload(self, architecture_id: str) -> None:
        """Unload the architecture from memory."""
        sys.modules.pop(self.module_name(architecture_id), None)

    def module_name(self, architecture_id: str) -> str:
        return f"_custom_{architecture_id}"

    def seed_builtins(self, db: Session) -> None:
        import os
        models_dir = os.path.join(os.path.dirname(__file__), "..", "models", "test")
        for name in ["ann", "lstm"]:
            path = os.path.join(models_dir, f"{name}.py")
            try:
                content = open(path, "rb").read()
                self.save_architecture(name, content, "system", db)
                logger.info(f"Seeded architecture: {name}")
            except ValueError:
                logger.debug(f"Architecture '{name}' already exists, skipping seed")
            except Exception as e:
                logger.warning(f"Failed to seed architecture '{name}': {e}")

    def list_architectures(self, db: Session) -> list[dict]:
        from src.db.architecture_config import ArchitectureConfigDB
        rows = db.query(ArchitectureConfigDB).all()
        return [
            {
                "name": r.architecture_id,
                "uploaded_by": r.uploaded_by,
                "uploaded_at": r.uploaded_at,
            }
            for r in rows
        ]

    def save_architecture(
        self, architecture_id: str, file: bytes, uploaded_by: str, db: Session
    ):
        """Validate and save the architecture to MinIO and record in DB."""
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
        )

        from src.db.architecture_config import ArchitectureConfigDB
        db.add(ArchitectureConfigDB(
            architecture_id=architecture_id,
            uploaded_by=uploaded_by,
            uploaded_at=datetime.now(),
        ))
        db.commit()

    def delete_architecture(self, architecture_id: str, db: Session):
        from src.db.architecture_config import ArchitectureConfigDB
        from src.db.model_config import ModelConfigDB

        in_use = db.query(ModelConfigDB).filter(
            ModelConfigDB.architecture == architecture_id
        ).first()
        if in_use:
            raise ValueError(f"Architecture '{architecture_id}' in use by one or more models")

        self.s3.delete_object(Bucket=_BUCKET_NAME, Key=architecture_id)

        row = db.get(ArchitectureConfigDB, architecture_id)
        if row:
            db.delete(row)
            db.commit()

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
        architecture_validations.dry_run.run(
            architecture_id, file, self._make_namespace
        )

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

    def _make_safe_import(self):

        allowed = architecture_validations.content.ALLOWED_IMPORTS

        def _safe_import(name, *args, **kwargs) -> dict:
            if name.split(".")[0] not in allowed:
                raise ImportError(f"Import not allowed: {name}")
            return __import__(name, *args, **kwargs)

        return _safe_import

    def _make_namespace(self, module_name: str) -> dict:
        return {
            "__name__": module_name,
            "__builtins__": {
                "__import__": self._make_safe_import(),
                "__build_class__": __build_class__,
                "float": float,
                "int": int,
                "str": str,
                "bool": bool,
                "list": list,
                "super": super,
                "dict": dict,
                "tuple": tuple,
                "len": len,
                "range": range,
                "None": None,
                "True": True,
                "False": False,
                "min": min,
                "max": max,
                "len": len,
                "range": range,
                "enumerate": enumerate,
                "zip": zip,
                "isinstance": isinstance,
                "print": print,
                "sum": sum,
                "abs": abs,
                "round": round,
            },
            "ModelInterface": ModelInterface,
        }
