"""Tests for utility modules."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from utils.file_io import (
    FileIOError,
    file_exists_and_not_empty,
    read_csv_file,
    read_json_file,
    read_parquet_file,
    write_json_file
)
from utils.llm_clients import (
    AnthropicClient,
    LLMClientError,
    LlamaClient,
    OpenAIClient,
    RateLimitedClient,
    create_llm_client
)
from utils.prompt_utils import (
    PromptGenerator,
    parse_demographic_text,
    truncate_text,
    validate_prompt_variables
)


class TestFileIO:
    """Test file I/O utilities."""
    
    def test_read_parquet_file_success(self, temp_dir):
        """Test successful parquet file reading."""
        # Create test parquet file
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        parquet_file = temp_dir / "test.parquet"
        df.to_parquet(parquet_file)
        
        result = read_parquet_file(parquet_file)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert list(result.columns) == ["col1", "col2"]
    
    def test_read_parquet_file_missing(self, temp_dir):
        """Test reading missing parquet file."""
        missing_file = temp_dir / "missing.parquet"
        
        with pytest.raises(FileIOError):
            read_parquet_file(missing_file)
    
    def test_read_csv_file_success(self, temp_dir):
        """Test successful CSV file reading."""
        csv_file = temp_dir / "test.csv"
        csv_file.write_text("col1,col2\n1,a\n2,b\n3,c")
        
        result = read_csv_file(csv_file)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert list(result.columns) == ["col1", "col2"]
    
    def test_write_json_file_success(self, temp_dir):
        """Test successful JSON file writing."""
        data = {"key": "value", "number": 42}
        json_file = temp_dir / "test.json"
        
        write_json_file(data, json_file)
        
        assert json_file.exists()
        with open(json_file) as f:
            loaded_data = json.load(f)
        
        assert loaded_data == data
    
    def test_read_json_file_success(self, temp_dir):
        """Test successful JSON file reading."""
        data = {"key": "value", "list": [1, 2, 3]}
        json_file = temp_dir / "test.json"
        
        with open(json_file, 'w') as f:
            json.dump(data, f)
        
        result = read_json_file(json_file)
        
        assert result == data
    
    def test_read_json_file_invalid(self, temp_dir):
        """Test reading invalid JSON file."""
        json_file = temp_dir / "invalid.json"
        json_file.write_text("invalid json content")
        
        with pytest.raises(FileIOError):
            read_json_file(json_file)
    
    def test_file_exists_and_not_empty(self, temp_dir):
        """Test file existence and emptiness check."""
        # Non-existent file
        assert not file_exists_and_not_empty(temp_dir / "missing.txt")
        
        # Empty file
        empty_file = temp_dir / "empty.txt"
        empty_file.touch()
        assert not file_exists_and_not_empty(empty_file)
        
        # File with content
        content_file = temp_dir / "content.txt"
        content_file.write_text("content")
        assert file_exists_and_not_empty(content_file)


class TestPromptUtils:
    """Test prompt generation utilities."""
    
    def test_parse_demographic_text_complete(self):
        """Test parsing complete demographic text."""
        text = "Age: 30, Gender: male, Education: PhD, Occupation: researcher, Location: Boston"
        
        demographics = parse_demographic_text(text)
        
        assert demographics["age"] == "30"
        assert demographics["gender"] == "male"
        assert demographics["education"] == "phd"
        assert demographics["occupation"] == "researcher"
        assert demographics["location"] == "boston"
    
    def test_parse_demographic_text_partial(self):
        """Test parsing partial demographic text."""
        text = "This person is 25 years old and works as a teacher"
        
        demographics = parse_demographic_text(text)
        
        assert demographics["age"] == "25"
        assert demographics["occupation"] == "teacher"
        assert demographics["gender"] == "unknown"
        assert demographics["education"] == "unknown"
        assert demographics["location"] == "unknown"
    
    def test_validate_prompt_variables_success(self):
        """Test prompt variable validation success."""
        prompt = "Hello {{name}}, your age is {{age}} and you live in {{location}}"
        variables = ["name", "age", "location"]
        
        result = validate_prompt_variables(prompt, variables)
        
        assert result is True
    
    def test_validate_prompt_variables_missing(self):
        """Test prompt variable validation with missing variables."""
        prompt = "Hello {{name}}, your age is {{age}}"
        variables = ["name", "age", "location"]  # location missing
        
        result = validate_prompt_variables(prompt, variables)
        
        assert result is False
    
    def test_truncate_text_no_truncation(self):
        """Test text truncation when no truncation needed."""
        text = "Short text"
        result = truncate_text(text, max_length=100)
        
        assert result == text
    
    def test_truncate_text_with_truncation(self):
        """Test text truncation when truncation needed."""
        text = "This is a very long text that needs to be truncated"
        result = truncate_text(text, max_length=20)
        
        assert len(result) <= 23  # 20 + "..." = 23
        assert result.endswith("...")
        assert not result.endswith(" ...")  # Should not cut mid-word
    
    def test_prompt_generator_initialization(self, temp_dir):
        """Test PromptGenerator initialization."""
        generator = PromptGenerator(temp_dir)
        
        assert generator.template_dir == temp_dir
        assert generator.env is not None
    
    def test_prompt_generator_render_template(self, temp_dir):
        """Test template rendering."""
        # Create test template
        template_file = temp_dir / "test.jinja"
        template_file.write_text("Hello {{name}}, you are {{age}} years old.")
        
        generator = PromptGenerator(temp_dir)
        context = {"name": "John", "age": 30}
        
        result = generator.render_template("test.jinja", context)
        
        assert result == "Hello John, you are 30 years old."
    
    def test_prompt_generator_missing_template(self, temp_dir):
        """Test rendering with missing template."""
        generator = PromptGenerator(temp_dir)
        
        with pytest.raises(Exception):  # PromptTemplateError
            generator.render_template("missing.jinja", {})
    
    def test_prompt_generator_missing_context_keys(self, temp_dir):
        """Test rendering with missing required context keys."""
        template_file = temp_dir / "test.jinja"
        template_file.write_text("Hello {{name}}")
        
        generator = PromptGenerator(temp_dir)
        context = {}  # Missing 'name'
        required_keys = ["name"]
        
        with pytest.raises(Exception):  # PromptTemplateError
            generator.render_template("test.jinja", context, required_keys)


class TestLLMClients:
    """Test LLM client implementations."""
    
    def test_create_llm_client_openai(self):
        """Test creating OpenAI client."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            client = create_llm_client("gpt-4")
            assert isinstance(client, OpenAIClient)
    
    def test_create_llm_client_anthropic(self):
        """Test creating Anthropic client."""
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}):
            client = create_llm_client("claude")
            assert isinstance(client, AnthropicClient)
    
    def test_create_llm_client_llama(self):
        """Test creating Llama client."""
        client = create_llm_client("llama")
        assert isinstance(client, LlamaClient)
    
    def test_create_llm_client_unsupported(self):
        """Test creating unsupported client."""
        with pytest.raises(LLMClientError):
            create_llm_client("unsupported-model")
    
    def test_openai_client_missing_api_key(self):
        """Test OpenAI client without API key."""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(LLMClientError):
                OpenAIClient()
    
    def test_anthropic_client_missing_api_key(self):
        """Test Anthropic client without API key."""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(LLMClientError):
                AnthropicClient()
    
    @patch('utils.llm_clients.OpenAI')
    def test_openai_client_generate_response(self, mock_openai_class, mock_openai_response):
        """Test OpenAI client response generation."""
        # Setup mock
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.return_value = mock_openai_response("4")
        
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            client = OpenAIClient()
            response = client.generate_response("Test prompt")
        
        assert response == "4"
        assert client.request_count == 1
        assert client.total_tokens > 0
    
    @patch('utils.llm_clients.Anthropic')
    def test_anthropic_client_generate_response(self, mock_anthropic_class, mock_anthropic_response):
        """Test Anthropic client response generation."""
        # Setup mock
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create.return_value = mock_anthropic_response("5")
        
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}):
            client = AnthropicClient()
            response = client.generate_response("Test prompt")
        
        assert response == "5"
        assert client.request_count == 1
        assert client.total_tokens > 0
    
    @patch('utils.llm_clients.requests.post')
    def test_llama_client_generate_response(self, mock_post):
        """Test Llama client response generation."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "3"}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        client = LlamaClient()
        response = client.generate_response("Test prompt")
        
        assert response == "3"
        assert client.request_count == 1
        assert client.total_tokens > 0
    
    def test_rate_limited_client(self):
        """Test rate limiting functionality."""
        # Create a mock client
        mock_client = MagicMock()
        mock_client.generate_response.return_value = "test response"
        mock_client.get_stats.return_value = {"request_count": 1}
        
        # Create rate limited client with high rate (to avoid long sleeps in tests)
        rate_limited = RateLimitedClient(mock_client, requests_per_minute=3600)  # 1 per second
        
        response = rate_limited.generate_response("test prompt")
        
        assert response == "test response"
        mock_client.generate_response.assert_called_once_with("test prompt")
    
    def test_client_stats_tracking(self):
        """Test client statistics tracking."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            client = OpenAIClient()
        
        # Initial stats
        stats = client.get_stats()
        assert stats["request_count"] == 0
        assert stats["total_tokens"] == 0
        assert stats["model_name"] == "gpt-4"
        
        # Reset stats
        client.reset_stats()
        stats = client.get_stats()
        assert stats["request_count"] == 0
        assert stats["total_tokens"] == 0