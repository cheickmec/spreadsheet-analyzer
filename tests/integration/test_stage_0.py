"""Test stage 0 directly."""

from pathlib import Path

from spreadsheet_analyzer.pipeline.stages.stage_0_integrity import stage_0_integrity_probe

# Test stage 0
test_file = Path("test_assets/generated/simple_test.xlsx")
result = stage_0_integrity_probe(test_file)

print(f"Result type: {type(result)}")
print(f"Result: {result}")

if hasattr(result, "value"):
    print("\nIntegrity details:")
    print(f"  Processing class: {result.value.processing_class}")
    print(f"  Is Excel: {result.value.is_excel}")
    print(f"  Is OOXML: {result.value.is_ooxml}")
    print(f"  Trust tier: {result.value.trust_tier}")
    print(f"  Validation passed: {result.value.validation_passed}")
    print(f"  File size: {result.value.metadata.size_bytes}")

if hasattr(result, "error"):
    print(f"\nError: {result.error}")
