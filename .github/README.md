# GitHub Configuration for Spreadsheet Analyzer

**Proprietary Software - Yiriden LLC**
**Owner**: Cheick Berthe (@cheickmec)

This directory contains GitHub-specific configuration files for the Spreadsheet Analyzer project.

## ğŸ“� Structure Overview

```
.github/
â”œâ”€â”€ workflows/                 # GitHub Actions CI/CD pipelines
â”   â”œâ”€â”€ ci.yml               # Main CI pipeline (lint, test, security, build)
â”   â”œâ”€â”€ performance.yml      # Performance testing and benchmarking
â”   â”œâ”€â”€ claude.yml           # Claude AI assistant integration
â”   â””â”€â”€ docs.yml             # Documentation building and deployment
â”
â”œâ”€â”€ ISSUE_TEMPLATE/           # Issue templates for different types
â”   â”œâ”€â”€ bug_report.md        # Bug report template
â”   â”œâ”€â”€ feature_request.md   # Feature request template
â”   â”œâ”€â”€ performance_issue.md # Performance issue template
â”   â”œâ”€â”€ documentation.md     # Documentation issue template
â”   â”œâ”€â”€ security_report.md   # Security vulnerability template
â”   â”œâ”€â”€ question.md          # Question/help template
â”   â””â”€â”€ config.yml           # Issue template configuration
â”
â”œâ”€â”€ pull_request_template.md  # PR template with comprehensive checklist
â”œâ”€â”€ dependabot.yml           # Automated dependency updates configuration
â”œâ”€â”€ CODEOWNERS              # Automatic reviewer assignment
â”œâ”€â”€ SECURITY.md             # Security policy and procedures
â”œâ”€â”€ .gitattributes          # Git attributes for file handling
â””â”€â”€ README.md               # This file
```

## ğŸš€ Workflows

### CI/CD Pipeline (`ci.yml`)

- **Triggers**: Push to main/develop, PRs
- **Jobs**:
  - Linting and type checking (Ruff, MyPy)
  - Unit and integration tests with coverage
  - Security scanning (Bandit, Safety, Trivy)
  - Docker image building and testing
  - Package building and publishing

### Performance Testing (`performance.yml`)

- **Triggers**: Weekly schedule, manual, PRs with 'performance' label
- **Features**:
  - Benchmarking against performance targets
  - Memory profiling
  - Regression detection
  - Performance report generation

### Claude Integration (`claude.yml`)

- **Triggers**: Comments mentioning @claude
- **Purpose**: AI-assisted code review and development

### Documentation (`docs.yml`)

- **Triggers**: Documentation changes
- **Features**:
  - Docstring coverage checking
  - API documentation generation
  - MkDocs site building
  - GitHub Pages deployment

## ğŸ”’ Security Features

1. **Dependency Scanning**: Automated via Dependabot
1. **Code Scanning**: Bandit for Python, Trivy for containers
1. **Security Policy**: Clear vulnerability reporting process
1. **CODEOWNERS**: Security team review for sensitive components

## ğŸ“‹ Issue Management

Templates provided for:

- Bug reports (with Excel-specific fields)
- Feature requests
- Performance issues
- Documentation improvements
- Security vulnerabilities
- General questions

## ğŸ¤� Contributing

The PR template ensures:

- Comprehensive testing
- Performance impact assessment
- Security considerations
- Documentation updates
- Code quality standards

## ğŸ”§ Configuration Notes

### Python/uv Specific

- All workflows use `uv` for package management
- Python 3.12 is the standard version
- Pre-commit hooks enforced in CI

### Performance Targets

Based on the system design document:

- File Upload (< 10MB): < 2 seconds
- Basic Analysis (< 10 sheets): < 5 seconds
- Deep AI Analysis (< 50 sheets): < 30 seconds

### Security Considerations

- No execution of VBA macros
- Sandboxed Jupyter kernel execution
- File size and type validation
- Resource limits enforced

## ğŸ“š References

- [Comprehensive System Design](../docs/design/comprehensive-system-design.md)
- [Deterministic Analysis Pipeline](../docs/design/deterministic-analysis-pipeline.md)
- [CLAUDE.md](../CLAUDE.md) - AI development guidelines

## ğŸ”„ Maintenance

- Weekly dependency updates via Dependabot
- Security updates prioritized (daily checks)
- Performance benchmarks run weekly
- Documentation auto-deployed on merge to main

______________________________________________________________________

*This is proprietary software owned by Yiriden LLC. All rights reserved.*
*For questions about GitHub configuration, please contact Cheick Berthe at cab25004@vt.edu.*
