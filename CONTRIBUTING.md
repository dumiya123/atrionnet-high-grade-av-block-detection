# Contributing to AtrionNet

Thank you for your interest in contributing to AtrionNet! This document provides guidelines and best practices for contributing to the project.

## 📋 Before You Start

Please familiarize yourself with the project by reading:
- [README.md](README.md) - Complete project documentation
- This file - Contribution guidelines

## 🛠️ How to Contribute

### 1. **Fork and Clone**
```bash
git clone https://github.com/dumiya123/atrionnet-high-grade-av-block-detection.git
cd atrionnet-high-grade-av-block-detection
```

### 2. **Create a Feature Branch**
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b bugfix/issue-description
```

### 3. **Set Up Development Environment**

**Backend (Python/ML):**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r ml_component/requirements.txt
```

**Frontend (React):**
```bash
cd frontend
npm install
npm run dev
```

### 4. **Make Your Changes**

**Python Code Style:**
- Follow PEP 8 guidelines
- Add docstrings to functions
- Include type hints
- Max line length: 100 characters

**Example:**
```python
def calculate_pr_interval(
    p_onset: int,
    r_peak: int,
    sampling_rate: int = 500
) -> float:
    """
    Calculate PR interval in milliseconds.
    
    Args:
        p_onset: P-wave onset sample index
        r_peak: R-peak sample index
        sampling_rate: ECG sampling frequency in Hz
    
    Returns:
        PR interval in milliseconds
    """
    pr_samples = r_peak - p_onset
    return (pr_samples / sampling_rate) * 1000
```

**JavaScript/React Code Style:**
- Follow ESLint configuration
- Use functional components
- Add JSDoc comments
- Proper error handling

### 5. **Test Your Changes**

**ML Component:**
```bash
# Run unit tests
python -m pytest ml_component/tests/ -v

# Run audit system
python backend/audit_system.py

# Test with sample data
python backend/test_xai_fix.py
```

**Frontend:**
```bash
cd frontend
npm run build
npm run preview
```

### 6. **Commit Your Changes**

Use conventional commit messages:
```
type(scope): subject

body (optional)
footer (optional)
```

**Types:** `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`

**Examples:**
```bash
git commit -m "feat(model): improve P-wave detection accuracy

- Implemented attention mechanism in encoder
- Added Gaussian heatmap smoothing
- Increased F1-score by 2.3% on test set

Closes #123"
```

```bash
git commit -m "fix(backend): handle missing API key gracefully"

git commit -m "docs(readme): update installation instructions"
```

### 7. **Push and Create Pull Request**

```bash
git push origin feature/your-feature-name
```

Then open a Pull Request on GitHub with:
- Clear title describing the change
- Detailed description of what was changed and why
- Reference to any related issues
- Screenshots/results if applicable

## 🐛 Reporting Bugs

Create an issue with:
- **Title:** Clear, concise description
- **Environment:** Python version, OS, GPU info
- **Steps to Reproduce:** Detailed reproduction steps
- **Expected vs Actual:** What should happen vs what happens
- **Attachments:** Test files, screenshots, error logs

**Example Bug Report:**
```
Title: P-wave detection fails on high-frequency noise

Environment:
- Python 3.10
- PyTorch 2.0.0
- CUDA 11.8
- Input: test_case_104.npy

Steps:
1. Load test_case_104.npy
2. Run inference with high baseline noise
3. Observe detection failure

Expected: Model should detect P-waves despite noise
Actual: All P-waves marked as false positives
```

## 💡 Feature Requests

Suggest improvements with:
- **Title:** Feature proposal
- **Description:** Detailed explanation
- **Use Case:** Why this feature is needed
- **Implementation Ideas:** Optional suggestions

**Example Feature Request:**
```
Title: Add real-time ECG stream processing

Description:
Enable processing of continuous ECG streams instead of fixed 10-second files.

Use Case:
Clinical monitoring systems need real-time processing for immediate alerts.

Implementation:
- Implement sliding window approach
- Buffer management for stream data
- Async processing with FastAPI WebSockets
```

## 📝 Code Review Guidelines

When reviewing PRs, check for:

**Code Quality:**
- ✅ Follows PEP 8 / ESLint rules
- ✅ Has proper docstrings
- ✅ No code duplication
- ✅ Clear variable names

**Functionality:**
- ✅ Tests pass
- ✅ No breaking changes
- ✅ Performance acceptable
- ✅ Handles edge cases

**Documentation:**
- ✅ Comments for complex logic
- ✅ Updated README if needed
- ✅ Docstrings complete

## 🧪 Testing

**Write Tests For:**
- New features
- Bug fixes
- Signal processing modifications
- API endpoints

**Test Coverage Target:** >80% on new code

```bash
# Run tests with coverage
python -m pytest ml_component/tests/ --cov=ml_component/src/ --cov-report=html

# View coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

## 📚 Documentation

When adding features or making changes:

1. **Update README.md** if adding major features
2. **Add docstrings** to all functions
3. **Include type hints** for parameters and returns
4. **Document API changes** in code comments
5. **Add examples** for complex functionality

**Example Docstring:**
```python
def detect_av_block(
    ecg_signal: np.ndarray,
    sampling_rate: int = 500,
    confidence_threshold: float = 0.45
) -> Dict[str, Any]:
    """
    Detect atrioventricular blocks in 12-lead ECG signal.
    
    This function uses the AtrionNet model to identify P-waves,
    QRS complexes, and T-waves, then classifies the rhythm
    based on PR intervals and conduction ratios.
    
    Args:
        ecg_signal: [12, 5000] array of ECG leads
        sampling_rate: Sampling frequency in Hz (default: 500)
        confidence_threshold: Model confidence threshold (default: 0.45)
    
    Returns:
        Dictionary containing:
        - 'diagnosis': str - Classification (Normal/1st/2nd/3rd degree)
        - 'confidence': float - Diagnostic confidence score
        - 'metrics': dict - Clinical metrics (PR, HR, etc.)
        - 'landmarks': dict - Detected wave boundaries
        - 'severity': str - Clinical severity assessment
    
    Raises:
        ValueError: If input shape is incorrect
        RuntimeError: If model cannot be loaded
    
    Example:
        >>> ecg = np.random.randn(12, 5000)
        >>> result = detect_av_block(ecg)
        >>> print(result['diagnosis'])
        '3rd Degree AV Block'
    """
```

## 🚀 Release Process

For maintainers:

1. **Update Version**
   - Update `__version__` in source code
   - Update version in `setup.py` or `pyproject.toml`

2. **Update Changelog**
   - Document new features
   - List bug fixes
   - Note breaking changes

3. **Tag Release**
   ```bash
   git tag -a v5.1.0 -m "AtrionNet v5.1.0 Release"
   git push origin v5.1.0
   ```

4. **Create GitHub Release**
   - Upload build artifacts
   - Add changelog
   - Mark as draft/prerelease if needed

## 📋 Pull Request Checklist

Before submitting, ensure:
- [ ] Code follows style guidelines
- [ ] New tests added/updated
- [ ] All tests pass
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
- [ ] Commit messages are clear
- [ ] No merge conflicts

## 🔄 Workflow Example

```bash
# 1. Create feature branch
git checkout -b feat/improved-p-wave-detection

# 2. Make changes and test
python -m pytest ml_component/tests/test_model.py

# 3. Commit with clear message
git commit -m "feat(model): implement multi-scale P-wave detection

- Added inception blocks with [9, 19, 39] kernel sizes
- Improved F1-score from 94.2% to 97.4%
- Added comprehensive test cases"

# 4. Push to remote
git push origin feat/improved-p-wave-detection

# 5. Create Pull Request on GitHub
# - Add description
# - Link related issues
# - Wait for review

# 6. Address feedback
git commit -m "fix: address code review feedback"
git push origin feat/improved-p-wave-detection

# 7. Merge when approved
```

## 🤝 Community Guidelines

- **Be Respectful:** Treat all contributors with respect
- **Be Constructive:** Provide helpful feedback with suggestions
- **Ask Questions:** Don't hesitate to ask for clarification
- **Share Knowledge:** Help others understand the codebase
- **Acknowledge Work:** Credit contributors appropriately

## 🔐 Security

When contributing:
- **Never commit credentials** (API keys, passwords, etc.)
- **Use `.gitignore`** for sensitive files
- **Report security issues privately** to: gamagedumindui@gmail.com
- **Don't hardcode** sensitive information

## ❓ Questions?

- **Documentation:** See [README.md](README.md)
- **Technical Issues:** Create GitHub Issue
- **General Questions:** Start GitHub Discussion
- **Direct Contact:** gamagedumindui@gmail.com

## ⚖️ License

By contributing to AtrionNet, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to AtrionNet! Your help makes this project better for everyone. 🙏**
