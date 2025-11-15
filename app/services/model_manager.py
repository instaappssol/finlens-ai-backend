"""
Model Manager Service

Handles saving and loading of trained ML models using joblib/pickle.
Supports both scikit-learn and PyTorch models.
Supports MongoDB GridFS for persistent storage across deployments.
"""

import os
import io
import joblib
import pickle
from pathlib import Path
from typing import Optional, Any, Dict
from datetime import datetime
import json
from gridfs import GridFS
from pymongo.database import Database

GRIDFS_AVAILABLE = True


class ModelManager:
    """Manages model persistence - save, load, and versioning"""

    def __init__(
        self,
        models_dir: str = "models",
        db: Optional[Database] = None,
        use_gridfs: bool = True,
    ):
        """
        Initialize ModelManager with storage directory.

        Args:
            models_dir: Directory path for storing model files (used for local cache)
            db: Optional MongoDB database connection for GridFS storage
            use_gridfs: If True and db is provided, use GridFS for persistent storage
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.metadata_file = self.models_dir / "model_metadata.json"

        # GridFS setup
        self.db = db
        self.use_gridfs = use_gridfs and db is not None and GRIDFS_AVAILABLE
        if self.use_gridfs:
            self.gridfs = GridFS(db, collection="models")
            # Also store metadata in MongoDB
            self.metadata_collection = db["model_metadata"]
        else:
            self.gridfs = None
            self.metadata_collection = None

    def _get_metadata(self) -> Dict[str, Any]:
        """Load model metadata from MongoDB or JSON file"""
        if self.use_gridfs and self.metadata_collection is not None:
            try:
                doc = self.metadata_collection.find_one({"_id": "model_metadata"})
                if doc:
                    return doc.get("metadata", {})
            except Exception as e:
                print(f"Error loading metadata from MongoDB: {e}, falling back to file")

        # Fallback to file
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_metadata(self, metadata: Dict[str, Any]):
        """Save model metadata to MongoDB and/or JSON file"""
        # Save to MongoDB if available
        if self.use_gridfs and self.metadata_collection is not None:
            try:
                self.metadata_collection.update_one(
                    {"_id": "model_metadata"},
                    {"$set": {"metadata": metadata, "updated_at": datetime.utcnow()}},
                    upsert=True,
                )
            except Exception as e:
                print(f"Error saving metadata to MongoDB: {e}, saving to file only")

        # Always save to file as backup
        try:
            with open(self.metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            print(f"Error saving metadata to file: {e}")

    def save_model(
        self,
        model: Any,
        model_name: str,
        version: str = "latest",
        metadata: Optional[Dict[str, Any]] = None,
        use_joblib: bool = True,
    ) -> str:
        """
        Save a trained model to GridFS (if available) and local disk cache.

        Args:
            model: The model object to save (scikit-learn, PyTorch, etc.)
            model_name: Name identifier for the model (e.g., "transaction_categorizer")
            version: Version string (default: "latest")
            metadata: Optional metadata dict (training date, accuracy, etc.)
            use_joblib: If True, use joblib (better for scikit-learn), else use pickle

        Returns:
            Path to saved model file (or GridFS filename)
        """
        # Create filename
        extension = ".joblib" if use_joblib else ".pkl"
        filename = f"{model_name}_{version}{extension}"
        filepath = self.models_dir / filename

        # Serialize model to bytes
        model_bytes = io.BytesIO()
        if use_joblib:
            joblib.dump(model, model_bytes)
        else:
            pickle.dump(model, model_bytes)
        model_bytes.seek(0)
        model_data = model_bytes.read()

        # Save to GridFS if available
        gridfs_filename = None
        if self.use_gridfs and self.gridfs is not None:
            try:
                gridfs_filename = f"{model_name}/{version}{extension}"
                # Delete existing file if it exists
                try:
                    existing = self.gridfs.find_one({"filename": gridfs_filename})
                    if existing is not None:
                        self.gridfs.delete(existing._id)
                except Exception:
                    pass

                # Save to GridFS
                self.gridfs.put(
                    model_data,
                    filename=gridfs_filename,
                    model_name=model_name,
                    version=version,
                    metadata=json.dumps(metadata or {}),
                    upload_date=datetime.utcnow(),
                )
                print(f"Model saved to GridFS: {gridfs_filename}")
            except Exception as e:
                print(f"Error saving to GridFS: {e}, saving to local only")

        # Always save to local disk as cache
        try:
            with open(filepath, "wb") as f:
                f.write(model_data)
        except Exception as e:
            print(f"Error saving to local cache: {e}")

        # Update metadata
        meta = self._get_metadata()
        if model_name not in meta:
            meta[model_name] = {}

        meta[model_name][version] = {
            "filepath": str(filepath),
            "gridfs_filename": gridfs_filename,
            "saved_at": datetime.utcnow().isoformat(),
            "version": version,
            "metadata": metadata or {},
            "stored_in_gridfs": self.use_gridfs and gridfs_filename is not None,
        }

        # Mark as latest if version is "latest"
        if version == "latest":
            meta[model_name]["current"] = version

        self._save_metadata(meta)

        return str(filepath) if filepath.exists() else (gridfs_filename or filename)

    def load_model(
        self,
        model_name: str,
        version: Optional[str] = None,
        use_joblib: bool = True,
    ) -> Any:
        """
        Load a saved model from GridFS (if available) or local disk.

        Args:
            model_name: Name identifier for the model
            version: Version to load (default: "latest" or "current")
            use_joblib: If True, use joblib, else use pickle

        Returns:
            Loaded model object

        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If model_name not found in metadata
        """
        meta = self._get_metadata()

        # If metadata is empty, try to load from GridFS directly
        if not meta and self.use_gridfs and self.gridfs is not None:
            try:
                # Try to find model in GridFS
                grid_file = self.gridfs.find_one(
                    {"model_name": model_name, "version": version or "latest"}
                )
                if grid_file is not None:
                    # Reconstruct metadata from GridFS
                    meta = {
                        model_name: {
                            version
                            or "latest": {
                                "gridfs_filename": grid_file.filename,
                                "saved_at": (
                                    grid_file.upload_date.isoformat()
                                    if hasattr(grid_file.upload_date, "isoformat")
                                    else str(grid_file.upload_date)
                                ),
                                "version": version or "latest",
                                "metadata": (
                                    json.loads(grid_file.metadata)
                                    if hasattr(grid_file, "metadata")
                                    and grid_file.metadata
                                    else {}
                                ),
                                "stored_in_gridfs": True,
                            },
                            "current": version or "latest",
                        }
                    }
                    self._save_metadata(meta)
            except Exception as e:
                print(f"Error loading metadata from GridFS: {e}")

        if model_name not in meta:
            raise ValueError(f"Model '{model_name}' not found in metadata")

        # Determine version
        if version is None:
            version = meta[model_name].get("current", "latest")

        if version not in meta[model_name]:
            raise ValueError(f"Version '{version}' not found for model '{model_name}'")

        version_info = meta[model_name][version]
        filepath = Path(version_info.get("filepath", ""))
        gridfs_filename = version_info.get("gridfs_filename")

        # Try to load from local cache first
        if filepath and filepath.exists():
            try:
                if use_joblib or filepath.suffix == ".joblib":
                    return joblib.load(filepath)
                else:
                    with open(filepath, "rb") as f:
                        return pickle.load(f)
            except Exception as e:
                print(f"Error loading from local cache: {e}, trying GridFS")

        # Load from GridFS if available
        if self.use_gridfs and self.gridfs is not None and gridfs_filename:
            try:
                grid_file = self.gridfs.find_one({"filename": gridfs_filename})
                if grid_file is not None:
                    model_data = grid_file.read()

                    # Save to local cache for next time
                    if filepath:
                        try:
                            with open(filepath, "wb") as f:
                                f.write(model_data)
                        except Exception:
                            pass

                    # Load from bytes
                    model_bytes = io.BytesIO(model_data)
                    if use_joblib or gridfs_filename.endswith(".joblib"):
                        return joblib.load(model_bytes)
                    else:
                        return pickle.load(model_bytes)
            except Exception as e:
                print(f"Error loading from GridFS: {e}")

        # If we get here, model not found
        raise FileNotFoundError(
            f"Model file not found: {filepath or gridfs_filename}. "
            f"Local cache: {filepath.exists() if filepath else False}, "
            f"GridFS: {gridfs_filename or 'N/A'}"
        )

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get information about a saved model.

        Args:
            model_name: Name identifier for the model

        Returns:
            Dictionary with model metadata
        """
        meta = self._get_metadata()
        if model_name not in meta:
            return {"error": f"Model '{model_name}' not found"}

        model_info = meta[model_name].copy()
        if "current" in model_info:
            current_version = model_info["current"]
            model_info["current_version_info"] = model_info.get(current_version, {})

        return model_info

    def list_models(self) -> list:
        """List all available models"""
        meta = self._get_metadata()
        return list(meta.keys())

    def delete_model(self, model_name: str, version: Optional[str] = None):
        """
        Delete a model file and its metadata from both GridFS and local cache.

        Args:
            model_name: Name identifier for the model
            version: Version to delete (default: deletes all versions)
        """
        meta = self._get_metadata()

        if model_name not in meta:
            raise ValueError(f"Model '{model_name}' not found")

        if version is None:
            # Delete all versions
            for v, info in meta[model_name].items():
                if v != "current":
                    # Delete from GridFS
                    if (
                        self.use_gridfs
                        and self.gridfs is not None
                        and info.get("gridfs_filename")
                    ):
                        try:
                            grid_file = self.gridfs.find_one(
                                {"filename": info["gridfs_filename"]}
                            )
                            if grid_file is not None:
                                self.gridfs.delete(grid_file._id)
                        except Exception as e:
                            print(f"Error deleting from GridFS: {e}")

                    # Delete local file
                    if "filepath" in info:
                        filepath = Path(info["filepath"])
                        if filepath.exists():
                            filepath.unlink()
            del meta[model_name]
        else:
            # Delete specific version
            if version in meta[model_name]:
                info = meta[model_name][version]

                # Delete from GridFS
                if (
                    self.use_gridfs
                    and self.gridfs is not None
                    and info.get("gridfs_filename")
                ):
                    try:
                        grid_file = self.gridfs.find_one(
                            {"filename": info["gridfs_filename"]}
                        )
                        if grid_file is not None:
                            self.gridfs.delete(grid_file._id)
                    except Exception as e:
                        print(f"Error deleting from GridFS: {e}")

                # Delete local file
                if "filepath" in info:
                    filepath = Path(info["filepath"])
                    if filepath.exists():
                        filepath.unlink()
                del meta[model_name][version]

        self._save_metadata(meta)
