## 📋 Description

**What changed?**

<!-- Describe your changes in detail -->

**Why was this change needed?**

<!-- Explain the problem this PR solves -->

**How was it implemented?**

<!-- Provide a high-level overview of your approach -->

## 📄 Type of Change

- [ ] 🐛 Bug fix (non-breaking change which fixes an issue)
- [ ] ✨ New feature (non-breaking change which adds functionality)
- [ ] 💥 Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] 📚 Documentation update
- [ ] ♻️ Refactoring (no functional changes)
- [ ] ⚡ Performance improvement
- [ ] ✅ Tests (adding missing tests or correcting existing tests)
- [ ] 🔧 Build/CI configuration
- [ ] 🎨 Style (formatting, missing semi-colons, etc; no code change)
- [ ] 🔒 Security fix

## 🧪 Testing

### Test Coverage

- [ ] Unit tests pass: `uv run pytest tests/unit`
- [ ] Integration tests pass: `uv run pytest tests/integration`
- [ ] Performance tests pass: `uv run pytest tests/performance`
- [ ] Coverage maintained/improved: Current: \_\_\_\_%

### Manual Testing

<!-- Describe the testing you've done -->

- [ ] Tested with small Excel files (< 1MB)
- [ ] Tested with medium Excel files (1-10MB)
- [ ] Tested with large Excel files (> 10MB)
- [ ] Tested with complex Excel features (formulas, charts, etc.)
- [ ] Tested error scenarios

## 📊 Performance Impact

<!-- How does this change affect performance? -->

- [ ] No performance impact
- [ ] Performance improvement (describe below)
- [ ] Potential performance regression (justified below)

### Performance Metrics

<!-- If applicable, provide before/after metrics -->

| Metric                    | Before | After | Change |
| ------------------------- | ------ | ----- | ------ |
| Analysis time (10 sheets) |        |       |        |
| Memory usage              |        |       |        |
| Token usage (AI)          |        |       |        |

## 🔒 Security Considerations

- [ ] Input validation added/updated
- [ ] No sensitive data logged
- [ ] Sandbox restrictions maintained
- [ ] Dependencies scanned for vulnerabilities
- [ ] Security tests added/updated

## 📚 Documentation

- [ ] Code includes appropriate comments
- [ ] Docstrings added/updated for new functions
- [ ] CLAUDE.md anchor comments added for complex logic
- [ ] API documentation updated if needed
- [ ] README updated if needed
- [ ] Architecture diagrams updated if needed

## ✅ Checklist

### Code Quality

- [ ] My code follows the project style guidelines
- [ ] I've run `uv run pre-commit run --all-files`
- [ ] I've run `uv run mypy src/`
- [ ] I've added type hints to all new functions
- [ ] No hardcoded values or magic numbers

### Testing

- [ ] I've added tests for new functionality
- [ ] All tests pass locally
- [ ] I've tested edge cases
- [ ] Integration tests cover the changes

### Dependencies

- [ ] No new dependencies added OR
- [ ] New dependencies are justified and documented
- [ ] Dependencies added with `uv add` (not pip)
- [ ] No security vulnerabilities in new dependencies

### Breaking Changes

<!-- If this PR introduces breaking changes -->

- [ ] Breaking changes are documented
- [ ] Migration guide provided
- [ ] Version bump planned

## 🔗 Related Issues

Closes #
Related to #

## 📸 Screenshots/Examples

<!-- If applicable, add screenshots or example outputs -->

<details>
<summary>Example Output</summary>

```
<!-- Show example analysis output or API responses -->
```

</details>

## 🚀 Deployment Notes

<!-- Any special deployment considerations -->

- [ ] No special deployment needed
- [ ] Environment variables added/changed: \_\_\_\_\_\_\_
- [ ] Configuration changes needed: \_\_\_\_\_\_\_
- [ ] Database migrations needed: \_\_\_\_\_\_\_

## 👀 Reviewer Guidelines

### Focus Areas

<!-- Highlight specific areas that need careful review -->

1.
1.
1.

### Testing Instructions

<!-- How can reviewers test this PR? -->

1.
1.
1.

______________________________________________________________________

**By submitting this PR, I confirm that:**

- I've read and followed the contribution guidelines
- I've tested my changes thoroughly
- I'm available to address review feedback
- I've signed the CLA (if required)
