"""Test stage 1 security scan directly."""

from pathlib import Path

from spreadsheet_analyzer.pipeline.stages.stage_1_security import stage_1_security_scan

# Test stage 1
test_file = Path("test_assets/generated/simple_test.xlsx")
result = stage_1_security_scan(test_file)

print(f"Result type: {type(result)}")
print(f"Result: {result}")

if hasattr(result, "value"):
    print("\nSecurity details:")
    print(f"  Risk level: {result.value.risk_level}")
    print(f"  Has macros: {result.value.has_macros}")
    print(f"  Has external links: {result.value.has_external_links}")
    print(f"  Threats: {len(result.value.threats)}")

    if result.value.threats:
        print("\nThreats found:")
        for threat in result.value.threats:
            print(f"  - {threat.threat_type}: {threat.description} (Risk: {threat.risk_level})")

if hasattr(result, "error"):
    print(f"\nError: {result.error}")
