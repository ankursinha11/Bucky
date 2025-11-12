"""
Multi-Provider AI Client
Supports both Azure OpenAI and AWS Bedrock for flexibility
"""

import os
import json
from typing import Dict, List, Optional, Any
from loguru import logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv()  # Load from .env
load_dotenv('.env.bedrock')  # Override with .env.bedrock if present


class AIProvider:
    """Base class for AI providers"""

    def __init__(self):
        self.enabled = False
        self.provider_name = "unknown"

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 4000
    ) -> str:
        """Send chat completion request and return response text"""
        raise NotImplementedError


class AzureOpenAIProvider(AIProvider):
    """Azure OpenAI provider"""

    def __init__(self):
        super().__init__()
        self.provider_name = "Azure OpenAI"

        try:
            from openai import AzureOpenAI

            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

            if not api_key or not endpoint:
                logger.warning("Azure OpenAI credentials not configured")
                return

            self.client = AzureOpenAI(
                azure_endpoint=endpoint,
                api_key=api_key,
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            )
            self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4")
            self.enabled = True
            logger.info(f"✓ {self.provider_name} initialized (deployment: {self.deployment_name})")

        except ImportError:
            logger.warning("OpenAI library not installed: pip install openai")
        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI: {e}")

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 4000
    ) -> str:
        """Send chat completion to Azure OpenAI"""
        if not self.enabled:
            return ""

        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Azure OpenAI API error: {e}")
            return ""


class AWSBedrockProvider(AIProvider):
    """AWS Bedrock provider (Claude models)"""

    def __init__(self):
        super().__init__()
        self.provider_name = "AWS Bedrock"

        try:
            import boto3

            access_key = os.getenv("AWS_ACCESS_KEY_ID")
            secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
            region = os.getenv("AWS_REGION", "us-east-1")

            if not access_key or not secret_key:
                logger.warning("AWS Bedrock credentials not configured")
                return

            self.client = boto3.client(
                service_name='bedrock-runtime',
                region_name=region,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key
            )

            self.model_id = os.getenv(
                "AWS_BEDROCK_MODEL",
                "anthropic.claude-3-sonnet-20240229-v1:0"
            )

            self.enabled = True
            logger.info(f"✓ {self.provider_name} initialized (model: {self.model_id})")

        except ImportError:
            logger.warning("boto3 not installed: pip install boto3")
        except Exception as e:
            logger.error(f"Failed to initialize AWS Bedrock: {e}")

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 4000
    ) -> str:
        """Send chat completion to AWS Bedrock (Claude)"""
        if not self.enabled:
            return ""

        try:
            # Convert OpenAI-style messages to Claude format
            # Claude uses separate system and messages
            system_message = ""
            conversation = []

            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    conversation.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })

            # Build Claude request
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": conversation
            }

            if system_message:
                request_body["system"] = system_message

            # Call Bedrock
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body)
            )

            # Parse response
            response_body = json.loads(response['body'].read())
            return response_body['content'][0]['text']

        except Exception as e:
            logger.error(f"AWS Bedrock API error: {e}")
            return ""


class MultiProviderAIClient:
    """
    Multi-provider AI client that auto-detects which provider to use

    Priority:
    1. Explicit AI_PROVIDER env var
    2. AWS Bedrock (if credentials present)
    3. Azure OpenAI (if credentials present)
    """

    def __init__(self):
        self.provider: Optional[AIProvider] = None
        self.enabled = False

        # Check which provider to use
        provider_choice = os.getenv("AI_PROVIDER", "").lower()

        if provider_choice == "bedrock":
            logger.info("🔧 Explicit provider selection: AWS Bedrock")
            self.provider = AWSBedrockProvider()
        elif provider_choice == "azure":
            logger.info("🔧 Explicit provider selection: Azure OpenAI")
            self.provider = AzureOpenAIProvider()
        else:
            # Auto-detect based on available credentials
            logger.info("🔍 Auto-detecting AI provider...")

            # Try AWS Bedrock first (for local testing)
            if os.getenv("AWS_ACCESS_KEY_ID"):
                self.provider = AWSBedrockProvider()
            # Fall back to Azure OpenAI
            elif os.getenv("AZURE_OPENAI_API_KEY"):
                self.provider = AzureOpenAIProvider()

        if self.provider and self.provider.enabled:
            self.enabled = True
            self.deployment_name = getattr(self.provider, 'deployment_name',
                                          getattr(self.provider, 'model_id', 'unknown'))
            logger.success(f"✅ AI Provider: {self.provider.provider_name}")
        else:
            logger.warning("⚠️ No AI provider configured - AI features disabled")
            logger.info("💡 Configure either:")
            logger.info("   - AWS Bedrock: Set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY in .env.bedrock")
            logger.info("   - Azure OpenAI: Set AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT in .env")

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 4000
    ) -> str:
        """Send chat completion request"""
        if not self.enabled or not self.provider:
            return ""

        return self.provider.chat_completion(messages, temperature, max_tokens)


# Convenience function for testing
def test_providers():
    """Test which AI providers are available"""
    print("=" * 60)
    print("AI PROVIDER TEST")
    print("=" * 60)

    # Test AWS Bedrock
    print("\n1. Testing AWS Bedrock...")
    bedrock = AWSBedrockProvider()
    if bedrock.enabled:
        print(f"   ✅ AWS Bedrock is available (model: {bedrock.model_id})")

        # Test request
        response = bedrock.chat_completion([
            {"role": "user", "content": "Say 'AWS Bedrock is working!' in exactly those words."}
        ])
        print(f"   Response: {response[:100]}...")
    else:
        print("   ❌ AWS Bedrock not available")

    # Test Azure OpenAI
    print("\n2. Testing Azure OpenAI...")
    azure = AzureOpenAIProvider()
    if azure.enabled:
        print(f"   ✅ Azure OpenAI is available (deployment: {azure.deployment_name})")

        # Test request
        response = azure.chat_completion([
            {"role": "user", "content": "Say 'Azure OpenAI is working!' in exactly those words."}
        ])
        print(f"   Response: {response[:100]}...")
    else:
        print("   ❌ Azure OpenAI not available")

    # Test auto-detection
    print("\n3. Testing auto-detection...")
    client = MultiProviderAIClient()
    if client.enabled:
        print(f"   ✅ Selected provider: {client.provider.provider_name}")
    else:
        print("   ❌ No provider available")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_providers()
