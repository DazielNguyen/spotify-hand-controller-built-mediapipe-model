# Hướng Dẫn Đóng Góp / Contributing Guide

## Tiếng Việt

Cảm ơn bạn đã quan tâm đến việc đóng góp cho dự án Spotify Hand Controller! Tài liệu này sẽ hướng dẫn bạn cách thiết lập và đóng góp vào dự án.

### 🚀 Bắt Đầu Nhanh

#### 1. Fork và Clone Repository

```bash
# Fork repository trên GitHub, sau đó clone về máy của bạn
git clone https://github.com/YOUR_USERNAME/spotify-hand-controller-built-mediapipe-model.git
cd spotify-hand-controller-built-mediapipe-model
```

#### 2. Thiết Lập Môi Trường Phát Triển

```bash
# Tạo virtual environment
python -m venv venv

# Kích hoạt virtual environment
# Trên macOS/Linux:
source venv/bin/activate
# Trên Windows:
# venv\Scripts\activate

# Cài đặt dependencies
pip install -r requirements.txt
```

#### 3. Cấu Hình Spotify API

```bash
# Copy file config mẫu
cp config/config.example.py config/config.py

# Chỉnh sửa config/config.py với credentials của bạn
# Lấy credentials tại: https://developer.spotify.com/dashboard
```

Điền thông tin vào `config/config.py`:
```python
spotify_client_id = "your_actual_client_id"
spotify_client_secret = "your_actual_client_secret"
spotify_redirect_uri = "http://localhost:8888/callback"
```

#### 4. Thiết Lập Remote Upstream

```bash
# Thêm remote upstream để đồng bộ với repo gốc
git remote add upstream https://github.com/ORIGINAL_OWNER/spotify-hand-controller-built-mediapipe-model.git

# Kiểm tra remotes
git remote -v
```

### 💻 Quy Trình Phát Triển

#### 1. Đồng Bộ Code Mới Nhất

Trước khi bắt đầu làm việc, luôn đồng bộ code từ repo gốc:

```bash
# Chuyển về nhánh main
git checkout main

# Lấy code mới nhất từ upstream
git fetch upstream

# Merge code mới vào nhánh main local
git merge upstream/main

# Push code mới lên fork của bạn
git push origin main
```

#### 2. Tạo Nhánh Mới

```bash
# Tạo nhánh mới cho feature/bugfix của bạn
git checkout -b feature/ten-tinh-nang

# Hoặc cho bugfix:
git checkout -b fix/ten-loi-can-sua
```

**Quy ước đặt tên nhánh:**
- `feature/` - Cho tính năng mới
- `fix/` - Cho sửa lỗi
- `docs/` - Cho cập nhật tài liệu
- `refactor/` - Cho refactoring code
- `test/` - Cho thêm/sửa tests

#### 3. Thực Hiện Thay Đổi

```bash
# Làm việc trên code của bạn
# ...

# Kiểm tra thay đổi
git status

# Thêm files đã thay đổi
git add .

# Commit với message rõ ràng
git commit -m "feat: thêm gesture nhận diện X Y Z"
```

**Quy ước commit message:**
- `feat:` - Thêm tính năng mới
- `fix:` - Sửa lỗi
- `docs:` - Cập nhật tài liệu
- `style:` - Format code, không thay đổi logic
- `refactor:` - Refactor code
- `test:` - Thêm/sửa tests
- `chore:` - Cập nhật dependencies, config, etc.

#### 4. Chạy Tests

```bash
# Chạy tất cả tests
pytest tests/

# Chạy test cụ thể
pytest tests/test_hand_detector.py

# Chạy với coverage
pytest --cov=src tests/
```

#### 5. Push Code

```bash
# Push nhánh của bạn lên fork
git push origin feature/ten-tinh-nang
```

#### 6. Tạo Pull Request

1. Truy cập repository gốc trên GitHub
2. Click nút "New Pull Request"
3. Chọn nhánh của bạn từ fork
4. Điền thông tin chi tiết về thay đổi
5. Submit Pull Request

### 📝 Hướng Dẫn Chi Tiết

#### Cấu Trúc Dự Án

- `data/` - Dữ liệu training (không commit dữ liệu lớn)
- `models/` - Models và checkpoints (không commit file models)
- `notebooks/` - Jupyter notebooks cho thử nghiệm
- `src/` - Source code chính của ứng dụng
- `training/` - Scripts để train model
- `tests/` - Unit tests và integration tests
- `config/` - Configuration files (không commit `config.py`)
- `docs/` - Tài liệu bổ sung

#### Viết Code

**Code Style:**
- Tuân thủ PEP 8 style guide cho Python
- Sử dụng meaningful variable names
- Thêm docstrings cho functions và classes
- Comment code phức tạp

**Example:**
```python
def detect_hand_gesture(frame, min_confidence=0.7):
    """
    Phát hiện cử chỉ tay trong frame.
    
    Args:
        frame: numpy array, hình ảnh input
        min_confidence: float, ngưỡng confidence tối thiểu
        
    Returns:
        dict: Thông tin về gesture được phát hiện
    """
    # Implementation...
    pass
```

#### Viết Tests

Mọi tính năng mới nên có tests:

```python
# tests/test_new_feature.py
import pytest
from src.new_feature import my_function

def test_my_function():
    """Test basic functionality."""
    result = my_function(input_data)
    assert result == expected_output

def test_my_function_edge_case():
    """Test edge case."""
    with pytest.raises(ValueError):
        my_function(invalid_input)
```

### 🎯 Gợi Ý Đóng Góp

#### Ý Tưởng Tính Năng Mới:
- Thêm gestures mới (peace sign, OK sign, etc.)
- Tích hợp với music players khác (Apple Music, YouTube Music)
- Thêm UI/Dashboard để monitor
- Cải thiện độ chính xác của model
- Thêm gesture customization trong runtime

#### Sửa Lỗi:
- Kiểm tra Issues tab trên GitHub
- Tìm issues được tag `good first issue` hoặc `help wanted`
- Báo cáo bugs mới mà bạn tìm thấy

#### Cải Thiện Tài Liệu:
- Thêm/cải thiện docstrings
- Viết tutorials
- Thêm examples
- Dịch tài liệu

### 🐛 Báo Cáo Lỗi

Khi báo cáo lỗi, vui lòng bao gồm:
1. Mô tả chi tiết vấn đề
2. Steps để reproduce
3. Expected behavior vs Actual behavior
4. Môi trường (OS, Python version, etc.)
5. Screenshots/logs nếu có

### 💡 Đề Xuất Tính Năng

Khi đề xuất tính năng mới:
1. Mô tả tính năng chi tiết
2. Giải thích use case
3. Đề xuất implementation (optional)
4. Mock-ups/examples (optional)

### 🔍 Code Review Process

1. Maintainer sẽ review Pull Request của bạn
2. Có thể có feedback/yêu cầu thay đổi
3. Thực hiện các thay đổi được yêu cầu
4. Sau khi approved, PR sẽ được merge

### 📞 Liên Hệ

- Mở Issue trên GitHub cho câu hỏi
- Tag maintainer trong comments nếu cần
- Tham gia discussions

### ✅ Checklist Trước Khi Submit PR

- [ ] Code chạy được và không có lỗi
- [ ] Đã chạy tests và tất cả pass
- [ ] Đã thêm tests cho code mới
- [ ] Đã cập nhật documentation nếu cần
- [ ] Code tuân thủ style guide
- [ ] Commit messages rõ ràng
- [ ] Đã đồng bộ với upstream/main mới nhất

---

## English

Thank you for your interest in contributing to the Spotify Hand Controller project! This document will guide you through the setup and contribution process.

### 🚀 Quick Start

#### 1. Fork and Clone Repository

```bash
# Fork the repository on GitHub, then clone to your machine
git clone https://github.com/YOUR_USERNAME/spotify-hand-controller-built-mediapipe-model.git
cd spotify-hand-controller-built-mediapipe-model
```

#### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### 3. Configure Spotify API

```bash
# Copy example config file
cp config/config.example.py config/config.py

# Edit config/config.py with your credentials
# Get credentials at: https://developer.spotify.com/dashboard
```

Fill in `config/config.py`:
```python
spotify_client_id = "your_actual_client_id"
spotify_client_secret = "your_actual_client_secret"
spotify_redirect_uri = "http://localhost:8888/callback"
```

#### 4. Set Up Upstream Remote

```bash
# Add upstream remote to sync with original repo
git remote add upstream https://github.com/ORIGINAL_OWNER/spotify-hand-controller-built-mediapipe-model.git

# Verify remotes
git remote -v
```

### 💻 Development Workflow

#### 1. Sync Latest Code

Before starting work, always sync code from original repo:

```bash
# Switch to main branch
git checkout main

# Fetch latest code from upstream
git fetch upstream

# Merge new code into local main
git merge upstream/main

# Push updates to your fork
git push origin main
```

#### 2. Create New Branch

```bash
# Create new branch for your feature/bugfix
git checkout -b feature/feature-name

# Or for bugfix:
git checkout -b fix/bug-name
```

**Branch naming conventions:**
- `feature/` - For new features
- `fix/` - For bug fixes
- `docs/` - For documentation updates
- `refactor/` - For code refactoring
- `test/` - For adding/fixing tests

#### 3. Make Changes

```bash
# Work on your code
# ...

# Check changes
git status

# Stage changed files
git add .

# Commit with clear message
git commit -m "feat: add X Y Z gesture recognition"
```

**Commit message conventions:**
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation update
- `style:` - Code formatting, no logic change
- `refactor:` - Code refactoring
- `test:` - Add/fix tests
- `chore:` - Update dependencies, config, etc.

#### 4. Run Tests

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_hand_detector.py

# Run with coverage
pytest --cov=src tests/
```

#### 5. Push Code

```bash
# Push your branch to fork
git push origin feature/feature-name
```

#### 6. Create Pull Request

1. Visit original repository on GitHub
2. Click "New Pull Request" button
3. Select your branch from fork
4. Fill in detailed information about changes
5. Submit Pull Request

### 📝 Detailed Guidelines

#### Project Structure

- `data/` - Training data (don't commit large data files)
- `models/` - Models and checkpoints (don't commit model files)
- `notebooks/` - Jupyter notebooks for experimentation
- `src/` - Main application source code
- `training/` - Scripts for model training
- `tests/` - Unit and integration tests
- `config/` - Configuration files (don't commit `config.py`)
- `docs/` - Additional documentation

#### Writing Code

**Code Style:**
- Follow PEP 8 style guide for Python
- Use meaningful variable names
- Add docstrings for functions and classes
- Comment complex code

**Example:**
```python
def detect_hand_gesture(frame, min_confidence=0.7):
    """
    Detect hand gesture in frame.
    
    Args:
        frame: numpy array, input image
        min_confidence: float, minimum confidence threshold
        
    Returns:
        dict: Information about detected gesture
    """
    # Implementation...
    pass
```

#### Writing Tests

Every new feature should have tests:

```python
# tests/test_new_feature.py
import pytest
from src.new_feature import my_function

def test_my_function():
    """Test basic functionality."""
    result = my_function(input_data)
    assert result == expected_output

def test_my_function_edge_case():
    """Test edge case."""
    with pytest.raises(ValueError):
        my_function(invalid_input)
```

### 🎯 Contribution Ideas

#### New Feature Ideas:
- Add new gestures (peace sign, OK sign, etc.)
- Integration with other music players (Apple Music, YouTube Music)
- Add UI/Dashboard for monitoring
- Improve model accuracy
- Add runtime gesture customization

#### Bug Fixes:
- Check Issues tab on GitHub
- Look for issues tagged `good first issue` or `help wanted`
- Report new bugs you find

#### Documentation Improvements:
- Add/improve docstrings
- Write tutorials
- Add examples
- Translate documentation

### 🐛 Reporting Bugs

When reporting bugs, please include:
1. Detailed problem description
2. Steps to reproduce
3. Expected behavior vs Actual behavior
4. Environment (OS, Python version, etc.)
5. Screenshots/logs if available

### 💡 Suggesting Features

When suggesting new features:
1. Detailed feature description
2. Explain use case
3. Suggest implementation (optional)
4. Mock-ups/examples (optional)

### 🔍 Code Review Process

1. Maintainer will review your Pull Request
2. May receive feedback/change requests
3. Implement requested changes
4. After approval, PR will be merged

### 📞 Contact

- Open Issue on GitHub for questions
- Tag maintainer in comments if needed
- Join discussions

### ✅ Pre-Submit PR Checklist

- [ ] Code runs without errors
- [ ] All tests pass
- [ ] Added tests for new code
- [ ] Updated documentation if needed
- [ ] Code follows style guide
- [ ] Clear commit messages
- [ ] Synced with latest upstream/main

---

## Thank You! / Cảm Ơn!

Your contributions make this project better for everyone! 🎉

Sự đóng góp của bạn làm cho dự án này tốt hơn cho mọi người! 🎉
