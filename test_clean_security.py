"""Test security scan on clean file."""

from pathlib import Path

from spreadsheet_analyzer.pipeline.stages.stage_1_security import stage_1_security_scan

# Test stage 1 on clean file
test_file = Path("test_data/clean_test.xlsx")
result = stage_1_security_scan(test_file)

print(f"Result type: {type(result)}")

if hasattr(result, "value"):
    print("\nSecurity details:")
    print(f"  Risk level: {result.value.risk_level}")
    print(f"  Risk score: {result.value.risk_score}")
    print(f"  Has macros: {result.value.has_macros}")
    print(f"  Has external links: {result.value.has_external_links}")
    print(f"  Is safe: {result.value.is_safe}")
    print(f"  Threats: {len(result.value.threats)}")

    if result.value.threats:
        print("\nThreats found:")
        for threat in result.value.threats:
            print(f"  - {threat.threat_type}: {threat.description}")
            print(f"    Location: {threat.location}")
            print(f"    Risk: {threat.risk_level} (severity: {threat.severity})")

if hasattr(result, "error"):
    print(f"\nError: {result.error}")
