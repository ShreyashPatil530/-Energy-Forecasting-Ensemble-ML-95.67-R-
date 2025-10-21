# Contributing to Energy Demand Forecasting

Thank you for your interest in contributing to this project! We welcome contributions from the community.

## How to Contribute

### 1. Fork the Repository
Click the "Fork" button on GitHub to create your own copy of the repository.

### 2. Clone Your Fork
```bash
git clone https://github.com/shreyashpatil/energy-forecasting.git
cd energy-forecasting
```

### 3. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
```

### 4. Make Your Changes
- Write clean, well-documented code
- Follow PEP 8 style guidelines
- Add comments for complex logic
- Update relevant documentation

### 5. Commit Your Changes
```bash
git commit -m "Add feature: description of changes"
```

### 6. Push to Your Branch
```bash
git push origin feature/your-feature-name
```

### 7. Open a Pull Request
- Clearly describe what your PR does
- Reference any related issues
- Include screenshots/results if applicable

## Code Style Guidelines

- Use Python 3.8+ compatible syntax
- Follow PEP 8 naming conventions
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Include type hints where possible

Example:
```python
def create_sequences(data: np.ndarray, seq_length: int = 24) -> tuple:
    """
    Create sequences for LSTM training.
    
    Parameters:
    -----------
    data : np.ndarray
        Input time series data
    seq_length : int
        Length of each sequence
        
    Returns:
    --------
    tuple
        (X sequences, y targets)
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)
```

## Types of Contributions

### Bug Reports
- Describe the bug clearly
- Include steps to reproduce
- Provide error messages or screenshots
- Mention your environment (OS, Python version, etc.)

### Feature Requests
- Explain the feature and its benefits
- Provide use cases
- Discuss implementation approach if you have ideas

### Documentation
- Fix typos and improve clarity
- Add examples and tutorials
- Update outdated information
- Improve docstrings

### Code Improvements
- Optimize performance
- Refactor complex code
- Add error handling
- Improve test coverage

## Testing

Before submitting a PR, please:

1. Test your changes locally
2. Run the full pipeline if modifying core code
3. Check that existing functionality still works
4. Test edge cases

## Documentation

- Update README.md for major changes
- Add docstrings to new functions
- Include comments for complex logic
- Update requirements.txt if adding dependencies

## Pull Request Process

1. Update the README.md with details of changes if applicable
2. Ensure all tests pass locally
3. Provide a clear description of your changes
4. Link any related issues
5. Your PR will be reviewed and merged after approval

## Code of Conduct

- Be respectful and constructive
- Focus on the idea, not the person
- Welcome diverse perspectives
- Help others learn and grow

## Questions?

- Open an issue for discussion
- Check existing issues and discussions
- Comment on related PRs
- Reach out respectfully

## Recognition

Contributors will be acknowledged in:
- README.md contributors section
- Release notes
- GitHub contributors page

Thank you for helping make this project better!
