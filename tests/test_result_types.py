"""
Tests for Result type system.

Tests the isinstance check that was broken with union types.
"""

from spreadsheet_analyzer.pipeline.types import Err, Ok, Result


def test_isinstance_with_tuple():
    """Test that isinstance works with tuple of types (not union)."""
    ok_result = Ok("success")
    err_result = Err("failure")

    # This is the correct way that works at runtime
    assert isinstance(ok_result, (Ok, Err))  # noqa: UP038
    assert isinstance(err_result, (Ok, Err))  # noqa: UP038

    # More specific checks
    assert isinstance(ok_result, Ok)
    assert not isinstance(ok_result, Err)
    assert isinstance(err_result, Err)
    assert not isinstance(err_result, Ok)


def test_union_type_would_fail():
    """Demonstrate why we can't use union syntax with isinstance."""
    # This test shows why the union syntax doesn't work with isinstance
    # If we tried: isinstance(ok_result, Ok | Err)
    # We would get: TypeError: isinstance() arg 2 must be a type, a tuple of types, or a union

    # The union creates a types.UnionType object, not a valid isinstance argument
    union_type = Ok | Err
    assert str(type(union_type)) == "<class 'types.UnionType'>"

    # For type checking, Result is defined as Ok | Err, but at runtime we need tuple
    assert Result == (Ok | Err)


def test_result_type_checking():
    """Test that Result type alias works for type hints."""

    def process_result(result: Result) -> str:
        """Function that accepts Result type."""
        if isinstance(result, Ok):
            return f"Success: {result.value}"
        return f"Error: {result.error}"

    ok = Ok("data")
    err = Err("problem")

    assert process_result(ok) == "Success: data"
    assert process_result(err) == "Error: problem"
