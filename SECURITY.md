# Security Policy

## 🔒 Supported Versions

We actively support the following versions of ScreenMonitorMCP with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | ✅ Yes             |
| < 1.0   | ❌ No              |

## 🚨 Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability in ScreenMonitorMCP, please report it responsibly.

### 📧 How to Report

**DO NOT** create a public GitHub issue for security vulnerabilities.

Instead, please:

1. **Email**: Send details to `security@screenmonitormcp.com` (if available)
2. **GitHub Security**: Use GitHub's private vulnerability reporting feature
3. **Direct Contact**: Contact the maintainers directly through GitHub

### 📋 What to Include

Please include the following information in your report:

- **Description**: Clear description of the vulnerability
- **Steps to Reproduce**: Detailed steps to reproduce the issue
- **Impact**: Potential impact and attack scenarios
- **Environment**: OS, Python version, dependencies
- **Proof of Concept**: If applicable (but avoid destructive examples)

### ⏱️ Response Timeline

- **Initial Response**: Within 48 hours
- **Assessment**: Within 1 week
- **Fix Development**: Depends on severity (1-4 weeks)
- **Public Disclosure**: After fix is released and users have time to update

## 🛡️ Security Considerations

### 🔑 API Keys and Secrets

ScreenMonitorMCP handles sensitive information:

- **OpenAI API Keys**: For AI analysis
- **Server API Keys**: For endpoint security
- **Screen Content**: Potentially sensitive screen data

**Best Practices:**
- Never commit `.env` files to version control
- Use environment variables for all secrets
- Rotate API keys regularly
- Limit API key permissions when possible

### 📸 Screen Data Privacy

ScreenMonitorMCP captures and analyzes screen content:

**Privacy Measures:**
- Screenshots are processed locally by default
- No screen data is sent to external services without explicit configuration
- Temporary files are cleaned up automatically
- User behavior data is stored locally

**User Responsibilities:**
- Be aware of what's on your screen when monitoring is active
- Configure AI providers carefully
- Review logs for sensitive information
- Use appropriate access controls

### 🌐 Network Security

**Outbound Connections:**
- AI provider APIs (OpenAI, OpenRouter, etc.)
- Package repositories (PyPI) during installation
- No other external connections by default

**Inbound Connections:**
- MCP server listens on localhost by default
- Configurable host/port settings
- API key authentication for endpoints

### 🔐 Authentication & Authorization

**Current Implementation:**
- API key-based authentication
- Local-only access by default
- No user management system

**Recommendations:**
- Use strong, unique API keys
- Restrict network access if needed
- Monitor access logs

## 🚫 Known Security Limitations

### Current Limitations

1. **No User Management**: Single-user system, no multi-user support
2. **Local Storage**: Behavior data stored in plain text JSON
3. **No Encryption**: Screen captures and logs are not encrypted at rest
4. **Process Permissions**: Requires screen capture permissions

### Planned Improvements

- Encrypted local storage options
- Enhanced authentication mechanisms
- Audit logging
- Permission management

## 🔧 Security Configuration

### Recommended Settings

```env
# Use strong API keys
API_KEY=your-strong-random-key-here

# Restrict network access
HOST=127.0.0.1  # localhost only
PORT=7777

# Disable debug logging in production
LOG_LEVEL=INFO
ENABLE_DEBUG_LOGGING=false

# Limit screenshot storage
SAVE_SCREENSHOTS=false  # if not needed
```

### Hardening Checklist

- [ ] Strong API keys configured
- [ ] Debug logging disabled
- [ ] Network access restricted
- [ ] Regular dependency updates
- [ ] Monitor access logs
- [ ] Review screen capture permissions
- [ ] Secure .env file permissions

## 🔄 Dependency Security

### Automated Scanning

We use automated tools to scan for vulnerabilities:

- **GitHub Dependabot**: Automatic dependency updates
- **Safety**: Python package vulnerability scanning
- **Bandit**: Static security analysis

### Manual Reviews

- Regular security reviews of dependencies
- Evaluation of new dependencies before addition
- Monitoring security advisories

## 📊 Security Metrics

We track security-related metrics:

- Time to patch vulnerabilities
- Dependency update frequency
- Security scan results
- Incident response times

## 🤝 Security Community

### Contributing to Security

- Report vulnerabilities responsibly
- Suggest security improvements
- Review security-related code changes
- Share security best practices

### Security Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Python Security Guidelines](https://python.org/dev/security/)
- [MCP Security Best Practices](https://modelcontextprotocol.io/security)

## 📞 Contact

For security-related questions or concerns:

- **Security Issues**: Use private reporting channels
- **General Security Questions**: Create a GitHub Discussion
- **Security Best Practices**: Check our documentation

---

**Remember**: Security is a shared responsibility. Please use ScreenMonitorMCP responsibly and follow security best practices.
