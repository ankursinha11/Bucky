#!/usr/bin/env python3
"""
Test Azure OpenAI Credentials
Quick test to verify your OpenAI API configuration
"""

import os
import sys
from pathlib import Path

print("=" * 70)
print("Azure OpenAI Credentials Test")
print("=" * 70)
print()

# Check if OpenAI is installed
try:
    from openai import AzureOpenAI
    print("✓ OpenAI library installed")
except ImportError:
    print("✗ OpenAI library not installed")
    print("  Install with: pip install openai")
    sys.exit(1)

# Check for .env file
env_file = Path(".env")
if env_file.exists():
    print("✓ .env file found")
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✓ .env file loaded")
    except ImportError:
        print("⚠️  python-dotenv not installed (optional)")
        print("  You can manually set environment variables")
else:
    print("⚠️  No .env file found")
    print("  You can set environment variables manually")

print()
print("Checking environment variables...")
print("-" * 70)

# Required environment variables
required_vars = {
    "AZURE_OPENAI_API_KEY": "Your Azure OpenAI API key",
    "AZURE_OPENAI_ENDPOINT": "Your Azure OpenAI endpoint (e.g., https://your-resource.openai.azure.com/)",
}

optional_vars = {
    "AZURE_OPENAI_API_VERSION": "API version (default: 2024-02-15-preview)",
    "AZURE_OPENAI_DEPLOYMENT_NAME": "Deployment name (default: gpt-4)",
}

missing_vars = []
found_vars = {}

for var, description in required_vars.items():
    value = os.getenv(var)
    if value:
        # Mask API key for security
        if "KEY" in var:
            display_value = value[:10] + "..." + value[-4:] if len(value) > 14 else "***"
        else:
            display_value = value
        print(f"✓ {var}: {display_value}")
        found_vars[var] = value
    else:
        print(f"✗ {var}: NOT SET")
        print(f"   Description: {description}")
        missing_vars.append(var)

print()
print("Optional variables:")
for var, description in optional_vars.items():
    value = os.getenv(var)
    if value:
        print(f"✓ {var}: {value}")
        found_vars[var] = value
    else:
        print(f"  {var}: Not set (will use default)")
        print(f"   Description: {description}")

print()
print("=" * 70)

if missing_vars:
    print("⚠️  MISSING REQUIRED VARIABLES")
    print("=" * 70)
    print()
    print("You need to set the following environment variables:")
    for var in missing_vars:
        print(f"  - {var}")
    print()
    print("How to set them:")
    print()
    print("Option 1 - Create .env file:")
    print("  Create a file named '.env' in the CodebaseIntelligence directory with:")
    print()
    print("  AZURE_OPENAI_API_KEY=your-api-key-here")
    print("  AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/")
    print("  AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4")
    print()
    print("Option 2 - Set in command line:")
    print("  Windows:")
    print("    set AZURE_OPENAI_API_KEY=your-api-key-here")
    print("    set AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/")
    print()
    print("  Mac/Linux:")
    print("    export AZURE_OPENAI_API_KEY=your-api-key-here")
    print("    export AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/")
    print()
    sys.exit(1)

# Try to connect
print("✓ ALL REQUIRED VARIABLES SET")
print("=" * 70)
print()
print("Testing connection to Azure OpenAI...")
print()

try:
    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
    )

    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4")

    print(f"Sending test request to deployment: {deployment_name}")

    response = client.chat.completions.create(
        model=deployment_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'Hello! I am working correctly.' if you can read this."}
        ],
        max_tokens=50,
        temperature=0,
    )

    answer = response.choices[0].message.content

    print()
    print("=" * 70)
    print("✓ SUCCESS! Azure OpenAI is working!")
    print("=" * 70)
    print()
    print("Response from OpenAI:")
    print(f"  {answer}")
    print()
    print("Configuration details:")
    print(f"  Endpoint: {os.getenv('AZURE_OPENAI_ENDPOINT')}")
    print(f"  Deployment: {deployment_name}")
    print(f"  Model: {response.model}")
    print(f"  API Version: {os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-15-preview')}")
    print()
    print("✓ Your Azure OpenAI credentials are working correctly!")
    print("✓ You can now use the chatbot in RAG mode with AI-powered answers")
    print()

except Exception as e:
    print()
    print("=" * 70)
    print("✗ ERROR: Connection failed")
    print("=" * 70)
    print()
    print(f"Error: {str(e)}")
    print()
    print("Common issues:")
    print("  1. Invalid API key")
    print("  2. Wrong endpoint URL")
    print("  3. Deployment name doesn't exist")
    print("  4. API version mismatch")
    print("  5. Network connectivity issues")
    print("  6. Insufficient permissions")
    print()
    print("Double-check:")
    print("  - API key is correct and active")
    print("  - Endpoint URL is correct (should end with /)")
    print("  - Deployment name matches your Azure OpenAI resource")
    print("  - Network allows access to Azure")
    print()
    sys.exit(1)
