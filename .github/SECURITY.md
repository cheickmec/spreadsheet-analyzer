# Security Policy

## 🔒 Overview

The Spreadsheet Analyzer is a proprietary product of Yiriden LLC that processes potentially sensitive Excel files and must maintain strict security standards. This document outlines our security policies, procedures, and best practices.

## 🛡️ Supported Versions

We provide security updates for the following versions:

| Version | Supported          | Notes                      |
| ------- | ------------------ | -------------------------- |
| 0.x.x   | :white_check_mark: | Active development         |
| < 0.1.0 | :x:                | Pre-release, not supported |

## 🚨 Reporting a Vulnerability

### For Critical Vulnerabilities

**DO NOT** create a public GitHub issue for critical security vulnerabilities.

Instead, please email: **security@yiriden.com**

Include:

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)
- Your contact information

We will acknowledge receipt within 48 hours and provide an initial assessment within 5 business days.

### For Low-Risk Issues

For non-critical security improvements, you may create a GitHub issue using the Security Report template.

## 🔐 Security Considerations

### Excel File Handling

1. **Input Validation**

   - All Excel files are validated before processing
   - File size limits are enforced
   - File type verification uses content inspection, not just extensions

1. **Macro Security**

   - VBA macros are NEVER executed
   - Macros are analyzed statically only
   - Files with macros are flagged for additional scrutiny

1. **Formula Execution**

   - Formulas are parsed but not evaluated with external data
   - External references are identified but not followed
   - Circular references are detected and handled safely

### Sandboxing

All code execution occurs within sandboxed Jupyter kernels with:

- No network access
- Limited filesystem access (whitelist-based)
- Resource limits (CPU, memory, execution time)
- Restricted module imports

### Data Protection

1. **In Transit**

   - All API communications use HTTPS
   - Files are encrypted during upload

1. **At Rest**

   - Temporary files are stored in secured directories
   - Files are automatically cleaned up after analysis
   - No permanent storage of user data without explicit consent

1. **In Memory**

   - Sensitive data is not logged
   - Memory is cleared after analysis
   - No data persists between analyses

### Dependencies

- All dependencies are regularly scanned for vulnerabilities
- Dependabot monitors for security updates
- Security updates are prioritized and fast-tracked

## 🛠️ Security Best Practices

### For Contributors

1. **Never commit sensitive data**

   - Use `.gitignore` for test files with real data
   - Sanitize all examples and test data
   - Use environment variables for credentials

1. **Input validation**

   - Always validate and sanitize user inputs
   - Use type hints and runtime validation
   - Implement proper error handling

1. **Code review**

   - All PRs require security-conscious review
   - Security-sensitive changes need additional review
   - Follow the principle of least privilege

### For Users

1. **File handling**

   - Only analyze files from trusted sources
   - Be cautious with files containing macros
   - Review analysis results before sharing

1. **API usage**

   - Keep API keys secure
   - Use appropriate authentication
   - Monitor usage for anomalies

1. **Deployment**

   - Follow deployment security guidelines
   - Keep the system updated
   - Monitor logs for suspicious activity

## 🔍 Security Features

### Built-in Protections

- **File quarantine**: Suspicious files are isolated
- **Analysis limits**: Prevents resource exhaustion
- **Audit logging**: All operations are logged
- **Error sanitization**: Error messages don't leak sensitive info

### Configuration Options

```yaml
security:
  max_file_size: 100MB
  allow_macros: false
  sandbox_timeout: 300s
  enable_audit_log: true
  quarantine_suspicious: true
```

## 📊 Security Audit Schedule

- **Weekly**: Dependency vulnerability scans
- **Monthly**: Code security review
- **Quarterly**: Full security audit
- **Annually**: Third-party security assessment

## 🚀 Security Roadmap

### Planned Enhancements

1. **Enhanced sandboxing** with gVisor
1. **File encryption** at rest
1. **Security scoring** for analyzed files
1. **Anomaly detection** in analysis patterns
1. **Zero-trust architecture** implementation

## 📞 Contact

- Security issues: security@yiriden.com
- General inquiries: contact@yiriden.com

______________________________________________________________________

*This is proprietary software owned by Yiriden LLC. All rights reserved.*
*Last updated: 2024-07-15*
