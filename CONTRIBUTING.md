# 🤝 Contributing to ScreenMonitorMCP

Thank you for your interest in contributing to **ScreenMonitorMCP**! This guide explains how you can get involved and contribute effectively.

---

## 🚀 Types of Contributions

### 🐛 Bug Reports
- Report bugs via [GitHub Issues](https://github.com/inkbytefo/ScreenMonitorMCP/issues)
- Include clear steps to reproduce
- Share your environment details (OS, Python version, etc.)

### 💡 Feature Requests
- Share new feature ideas in Issues
- Explain why the feature is useful
- Add example use cases if possible

### 🔧 Code Contributions
- Fork the repository and create a feature branch
- Follow the coding standards below
- Add relevant tests
- Submit a Pull Request (PR)

---

## 🛠️ Development Setup

### 1. Clone the Repository

```bash
git clone https://github.com/inkbytefo/ScreenMonitorMCP.git
cd ScreenMonitorMCP
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables

```bash
cp .env.example .env
# Edit the .env file with your local settings
```

---

## 🧑‍💻 Code Standards

### Python Style
- Follow **PEP 8**
- Use **type hints**
- Add **docstrings** and meaningful variable names

### Commit Message Format

```
feat: add new UI detection algorithm
fix: resolve OCR encoding issue
docs: update installation guide
test: add unit tests for monitoring
```

### Branch Naming

```
feature/ui-detection-enhancement
bugfix/ocr-unicode-error
docs/contributing-guide
```

---

## 🦪 Testing Guidelines

### Run Unit Tests

```bash
# Run all tests
python -m pytest

# Run a specific test file
python test_revolutionary_features.py
```

### Manual Testing

```bash
# Start the main server
python main.py

# Connect using a MCP-compatible client (e.g., Claude Desktop)
```

---

## 📋 Pull Request Process

1. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Your Changes**
   - Write code
   - Add/Update tests
   - Update docs if necessary

3. **Commit and Push**
   ```bash
   git add .
   git commit -m "feat: describe your feature"
   git push origin feature/your-feature-name
   ```

4. **Open a Pull Request**
   - Describe what you changed
   - Link to any related issue(s)

---

## 🎯 Contribution Focus Areas

### 🔥 Priority Areas
- **UI Detection**: New detection algorithms
- **OCR Improvements**: Better text recognition
- **Performance**: Optimize system performance
- **Cross-Platform Support**: Improve Linux/macOS compatibility
- **Documentation**: Improve English guides and usage examples

### 🧠 AI/ML Contributions
- Behavior prediction models
- Smart detection enhancements
- Optimizing computer vision modules

### 🔧 Infrastructure
- CI/CD improvements
- Dockerization
- Dependency/package management

---

## 📚 Documentation Standards

### What to Document
- New features: how to use them
- Setup and install instructions
- Troubleshooting guidance

### In Code
- Use **docstrings**
- Add **type hints**
- Comment complex logic inline

---

## 🐛 Bug Fix Workflow

1. **Create an Issue**
   - Describe the bug clearly
   - Share reproduction steps
   - Explain expected vs actual behavior

2. **Develop the Fix**
   - Keep changes minimal and clear
   - Cover edge cases

3. **Test It**
   - Add unit tests
   - Manually verify the fix
   - Check for regressions

---

## 🔒 Security Best Practices

### Reporting Security Issues
- Do **not** post public issues for vulnerabilities
- Report them privately via:  
  📧 `security@screenmonitormcp.com`

### Secrets & Keys
- Never commit `.env` files or API secrets
- Check `.gitignore` includes sensitive files

---

## 📲 Communication

### GitHub
- **Issues**: Bug reports, feature requests
- **Discussions**: General ideas and questions
- **Pull Requests**: Code changes and reviews

### Community
- Discord: *Coming Soon*
- Twitter: *Coming Soon*

---

## 🏆 Contributors

All contributors will be listed in the README and credited for their work. Thank you for making ScreenMonitorMCP better!

---

**🚀 Let’s build something revolutionary together!**