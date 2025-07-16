"""Test fixture management system for expected outputs."""

import hashlib
import json
from dataclasses import asdict, is_dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any

try:
    import jsonschema

    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False


class FixtureNotFoundError(FileNotFoundError):
    """Raised when a fixture file is not found."""

    def __init__(self, fixture_path: Path) -> None:
        """Initialize the error with fixture path."""
        super().__init__(f"Fixture not found: {fixture_path}\nRun with --update-fixtures to create it.")


class FixtureEncoder(json.JSONEncoder):
    """Custom JSON encoder for test fixtures that handles complex types."""

    def default(self, obj):
        # Handle datetime objects
        if isinstance(obj, datetime):
            return {"__type__": "datetime", "value": obj.isoformat()}

        if isinstance(obj, date):
            return {"__type__": "date", "value": obj.isoformat()}

        # Handle Decimal
        if isinstance(obj, Decimal):
            return {"__type__": "Decimal", "value": str(obj)}

        # Handle dataclasses - just convert to dict without type markers
        if is_dataclass(obj) and not isinstance(obj, type):
            return asdict(obj)

        # Handle Enums
        if isinstance(obj, Enum):
            return {"__type__": "Enum", "__class__": obj.__class__.__name__, "value": obj.value}

        # Handle sets and frozensets
        if isinstance(obj, set | frozenset):
            return {"__type__": "frozenset" if isinstance(obj, frozenset) else "set", "data": list(obj)}

        # Handle Path objects
        if isinstance(obj, Path):
            return {"__type__": "Path", "value": str(obj)}

        return super().default(obj)


def decode_fixture(dct: dict[str, Any]) -> Any:
    """Decode special types from JSON fixture data."""
    if "__type__" not in dct:
        return dct

    obj_type = dct["__type__"]

    if obj_type == "datetime":
        return datetime.fromisoformat(dct["value"])

    if obj_type == "date":
        return date.fromisoformat(dct["value"])

    if obj_type == "Decimal":
        return Decimal(dct["value"])

    if obj_type == "set":
        return set(dct["data"])

    if obj_type == "frozenset":
        return frozenset(dct["data"])

    if obj_type == "Path":
        return Path(dct["value"])

    # For dataclasses and Enums, return the dict representation
    # The test can reconstruct if needed
    return dct


class FixtureManager:
    """Manages test fixtures for expected outputs."""

    def __init__(self, base_path: Path | None = None):
        if base_path is None:
            # Default to tests/fixtures in the project root
            self.base_path = Path(__file__).parent.parent.parent.parent / "tests" / "fixtures"
        else:
            self.base_path = Path(base_path)

        self.outputs_dir = self.base_path / "outputs"
        self.inputs_dir = self.base_path / "inputs"
        self.schemas_dir = self.base_path / "schemas"

        # Create directories if they don't exist
        for dir_path in [self.outputs_dir, self.inputs_dir, self.schemas_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def _get_fixture_path(self, test_module: str, test_name: str) -> Path:
        """Get the path for a fixture file."""
        # Remove 'test_' prefix and '.py' suffix if present
        test_module = test_module.removeprefix("test_")
        test_module = test_module.removesuffix(".py")

        fixture_dir = self.outputs_dir / test_module
        fixture_dir.mkdir(exist_ok=True)

        return fixture_dir / f"{test_name}.json"

    def load(self, test_module: str, test_name: str) -> Any:
        """Load expected output from fixture file."""
        fixture_path = self._get_fixture_path(test_module, test_name)

        if not fixture_path.exists():
            raise FixtureNotFoundError(fixture_path)

        with fixture_path.open() as f:
            return json.load(f, object_hook=decode_fixture)

    def save(self, test_module: str, test_name: str, data: Any) -> None:
        """Save output to fixture file."""
        fixture_path = self._get_fixture_path(test_module, test_name)

        # Create metadata
        metadata = {
            "_metadata": {
                "created": datetime.now(tz=UTC).isoformat(),
                "test_module": test_module,
                "test_name": test_name,
                "data_hash": self._compute_hash(data),
            },
            "data": data,
        }

        with fixture_path.open("w") as f:
            json.dump(metadata, f, indent=2, cls=FixtureEncoder, sort_keys=True)

    def update(self, test_module: str, test_name: str, data: Any) -> dict[str, Any]:
        """Update fixture and return diff information."""
        fixture_path = self._get_fixture_path(test_module, test_name)

        old_data = None
        if fixture_path.exists():
            old_data = self.load(test_module, test_name)

        self.save(test_module, test_name, data)

        return {
            "path": str(fixture_path),
            "existed": old_data is not None,
            "old_hash": self._compute_hash(old_data) if old_data else None,
            "new_hash": self._compute_hash(data),
        }

    def _compute_hash(self, data: Any) -> str:
        """Compute a hash of the data for change detection."""
        json_str = json.dumps(data, cls=FixtureEncoder, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()[:8]

    def list_fixtures(self, test_module: str | None = None) -> dict[str, list]:
        """List all available fixtures."""
        fixtures = {}

        modules = [test_module] if test_module else [d.name for d in self.outputs_dir.iterdir() if d.is_dir()]

        for module in modules:
            module_dir = self.outputs_dir / module
            if module_dir.exists():
                fixtures[module] = [f.stem for f in module_dir.glob("*.json")]

        return fixtures

    def validate_against_schema(self, data: Any, schema_name: str) -> bool:
        """Validate data against a JSON schema."""
        if not HAS_JSONSCHEMA:
            print("jsonschema not installed, skipping validation")
            return True

        schema_path = self.schemas_dir / f"{schema_name}.schema.json"
        if not schema_path.exists():
            return True  # No schema, assume valid

        with schema_path.open() as f:
            schema = json.load(f)

        try:
            jsonschema.validate(data, schema)
        except jsonschema.ValidationError as e:
            print(f"Schema validation failed: {e}")
            return False
        else:
            return True


# Global fixture manager instance
_fixture_manager = FixtureManager()


def load_expected_output(test_module: str, test_name: str) -> Any:
    """Load expected output for a test."""
    return _fixture_manager.load(test_module, test_name)


def save_expected_output(test_module: str, test_name: str, data: Any) -> None:
    """Save expected output for a test."""
    _fixture_manager.save(test_module, test_name, data)


def update_expected_output(test_module: str, test_name: str, data: Any) -> dict[str, Any]:
    """Update expected output and return change information."""
    return _fixture_manager.update(test_module, test_name, data)
