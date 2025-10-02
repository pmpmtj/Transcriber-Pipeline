# Transcribe Pipeline Testing Framework

This directory contains comprehensive unit tests for the transcribe pipeline, built with a dependency injection framework for effective testing.

## Overview

The testing framework provides:
- **Dependency Injection**: All external dependencies are abstracted and mockable
- **Unit Tests**: Comprehensive coverage of individual functions and classes
- **Mock Implementations**: Controlled testing without external dependencies
- **Test Fixtures**: Reusable test data and configurations
- **Coverage Reporting**: Track test coverage and identify gaps

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── run_tests.py             # Test runner script
├── unit/                    # Unit tests by module
│   ├── test_config.py       # Configuration tests
│   ├── test_ffprobe.py      # Audio analysis tests
│   ├── test_transcriber.py  # Transcription tests
│   └── test_segmenter.py    # Audio segmentation tests (TODO)
├── integration/             # Integration tests (TODO)
└── fixtures/                # Test data and samples (TODO)
```

## Running Tests

### Quick Start

```bash
# Run all tests
python tests/run_tests.py

# Run with verbose output
python tests/run_tests.py -v

# Run specific test file
python tests/run_tests.py --pattern test_config.py

# Run with coverage report
python tests/run_tests.py --coverage
```

### Using pytest directly

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/unit/test_config.py

# Run with coverage
pytest tests/ --cov=src.transcribe_pipeline --cov-report=html

# Run with verbose output
pytest tests/ -v
```

## Dependency Injection Framework

### Interfaces

The framework defines clear interfaces for all external dependencies:

- **SubprocessProvider**: FFmpeg/FFprobe operations
- **OpenAIClientProvider**: OpenAI API interactions
- **FileSystemProvider**: File I/O operations
- **TimeProvider**: Time-related functions
- **EnvironmentProvider**: Environment variable access

### Mock Implementations

Each interface has a corresponding mock implementation for testing:

- **MockSubprocessProvider**: Records commands and returns configurable results
- **MockOpenAIClientProvider**: Simulates API responses and errors
- **MockFileSystemProvider**: In-memory file system simulation
- **MockTimeProvider**: Controlled time operations
- **MockEnvironmentProvider**: Configurable environment variables

### Usage in Tests

```python
def test_transcription(mock_openai_provider, mock_fs_provider):
    # Setup mock responses
    mock_openai_provider.set_transcription_result(
        Path("audio.m4a"), 
        {"text": "Mock transcription"}
    )
    
    # Run function with mocked dependencies
    result = transcribe_audio(
        Path("audio.m4a"), 
        openai_provider=mock_openai_provider,
        fs_provider=mock_fs_provider
    )
    
    # Verify results
    assert result["text"] == "Mock transcription"
```

## Test Coverage

Current coverage targets:
- **Configuration Module**: 95%+ coverage
- **Audio Analysis (FFprobe)**: 90%+ coverage  
- **Transcription Module**: 85%+ coverage
- **Overall Target**: 80%+ coverage

## Test Categories

### Unit Tests
- **Configuration Tests**: Validation, CLI overrides, edge cases
- **Audio Analysis Tests**: FFprobe integration, error handling
- **Transcription Tests**: OpenAI API integration, retry logic
- **Utility Tests**: Path handling, logging, file operations

### Integration Tests (Planned)
- **End-to-end Pipeline**: Full transcription workflow
- **CLI Integration**: Command-line interface testing
- **File System Integration**: Real file operations with cleanup

## Test Fixtures

### Common Fixtures

- `sample_config`: Default configuration for testing
- `sample_audio_metadata`: Mock audio file metadata
- `sample_manifest`: Sample transcription manifest
- `mock_*_provider`: Mock implementations for all dependencies
- `temp_dir`: Temporary directory for test files

### Custom Fixtures

```python
@pytest.fixture
def custom_config():
    """Custom configuration for specific tests."""
    config = create_default_config()
    config.model.max_retries = 1
    return config
```

## Writing New Tests

### 1. Test Structure

```python
class TestModuleName:
    """Test class for specific module."""
    
    def test_function_success(self, mock_provider):
        """Test successful function execution."""
        # Arrange
        setup_test_data()
        
        # Act
        result = function_under_test()
        
        # Assert
        assert result == expected_value
```

### 2. Mock Setup

```python
def test_with_mock(mock_openai_provider):
    # Setup expected behavior
    mock_openai_provider.set_transcription_result(
        Path("test.m4a"),
        {"text": "Expected result"}
    )
    
    # Test the function
    result = transcribe_file(Path("test.m4a"))
    
    # Verify interactions
    assert len(mock_openai_provider.clients_created) == 1
```

### 3. Error Testing

```python
def test_error_handling(mock_provider):
    # Setup error condition
    mock_provider.set_error(Path("test.m4a"), Exception("Test error"))
    
    # Test error handling
    with pytest.raises(Exception, match="Test error"):
        function_that_should_fail()
```

## Continuous Integration

The test framework is designed to work with CI/CD systems:

- **GitHub Actions**: Automated testing on push/PR
- **Coverage Reporting**: HTML and terminal coverage reports
- **Test Isolation**: Each test is independent and can run in parallel
- **Mock Dependencies**: No external dependencies required for testing

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure the project is installed in development mode:
   ```bash
   pip install -e .
   ```

2. **Missing Dependencies**: Install test dependencies:
   ```bash
   pip install pytest pytest-cov
   ```

3. **Path Issues**: Tests should be run from the project root directory

### Debug Mode

Run tests with maximum verbosity for debugging:

```bash
pytest tests/ -vvv --tb=long
```

## Future Enhancements

- **Integration Tests**: Full pipeline testing with real files
- **Performance Tests**: Benchmarking and load testing
- **Property-Based Testing**: Hypothesis-based test generation
- **Mutation Testing**: Test quality validation
- **Visual Testing**: CLI output validation

## Contributing

When adding new features:

1. **Write Tests First**: Follow TDD principles
2. **Use Dependency Injection**: Make code testable from the start
3. **Add Fixtures**: Create reusable test data
4. **Document Tests**: Clear test names and docstrings
5. **Maintain Coverage**: Keep coverage above target thresholds
