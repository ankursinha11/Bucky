# Azure OpenAI Setup Guide

## Quick Setup

### 1. Create .env file from template
```bash
cp .env.example .env
```

### 2. Edit .env file with your Azure OpenAI credentials

Open `.env` file and update these lines:

```bash
# Required: Your Azure OpenAI credentials
AZURE_OPENAI_ENDPOINT=https://YOUR-RESOURCE-NAME.openai.azure.com/
AZURE_OPENAI_API_KEY=YOUR-API-KEY-HERE
AZURE_OPENAI_DEPLOYMENT_NAME=YOUR-GPT4-DEPLOYMENT-NAME
AZURE_OPENAI_API_VERSION=2024-02-15-preview

# Optional: Use Azure Text Embedding (or leave commented to use FREE local embeddings)
# AZURE_OPENAI_EMBEDDING_DEPLOYMENT=YOUR-EMBEDDING-DEPLOYMENT-NAME
```

### 3. Find Your Azure OpenAI Values

**Azure OpenAI Endpoint:**
- Go to Azure Portal → Your OpenAI Resource → Keys and Endpoint
- Copy the "Endpoint" value (e.g., `https://my-openai.openai.azure.com/`)

**API Key:**
- Same location → Copy "KEY 1" or "KEY 2"

**Deployment Name:**
- Go to Azure OpenAI Studio → Deployments
- Copy the deployment name of your GPT-4 model (NOT the model name)
- Example: If you deployed gpt-4 with name "gpt4-deployment", use "gpt4-deployment"

**Embedding Deployment (Optional):**
- If you deployed a text embedding model (text-embedding-ada-002), copy its deployment name
- If not configured, the system will use FREE local embeddings (sentence-transformers)
- **Recommendation:** Use FREE local embeddings to save costs!

## Example .env Configuration

```bash
# Real example (with fake credentials)
AZURE_OPENAI_ENDPOINT=https://mycompany-openai.openai.azure.com/
AZURE_OPENAI_API_KEY=abc123def456ghi789jkl012mno345pqr678stu
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4-deployment
AZURE_OPENAI_API_VERSION=2024-02-15-preview

# Using FREE local embeddings (recommended)
# AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-deployment
```

## Test Your Configuration

### Test 1: Verify credentials are loaded
```bash
python3 -c "
import os
from dotenv import load_dotenv
load_dotenv()

print('Azure OpenAI Endpoint:', os.getenv('AZURE_OPENAI_ENDPOINT'))
print('API Key Set:', 'Yes' if os.getenv('AZURE_OPENAI_API_KEY') else 'No')
print('Deployment Name:', os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'))
"
```

### Test 2: Index and query the chatbot
```bash
# Index your codebase
python3 index_codebase.py --abinitio-path "/path/to/your/files"

# Start chatbot - should show "RAG mode enabled"
python3 chatbot_cli.py
```

You should see:
```
✓ Azure OpenAI API key found - RAG mode enabled
```

### Test 3: Ask a question
```
You: What processes are in the codebase?
```

If configured correctly, you'll get an AI-generated answer instead of just search results!

## Troubleshooting

### Error: "Invalid API key"
- Check your API key is correct (no extra spaces)
- Make sure you're using KEY 1 or KEY 2 from Azure Portal

### Error: "Deployment not found"
- Verify the deployment name matches exactly (case-sensitive!)
- Check Azure OpenAI Studio → Deployments for the correct name

### Error: "Module 'dotenv' not found"
- Install: `pip install python-dotenv`

### Chatbot still in "search-only mode"
- Make sure `.env` file exists in the project root
- Verify `AZURE_OPENAI_API_KEY` is set in `.env`
- Check `.env` is not in `.gitignore` (wait, it should be!)
- Try setting environment variables directly: `export AZURE_OPENAI_API_KEY="..."`

## Cost Considerations

**With Azure OpenAI (RAG Mode):**
- Embedding: FREE (using local sentence-transformers)
- GPT-4 queries: ~$0.01-0.03 per query
- 100 queries ≈ $1-3

**Search-Only Mode (No OpenAI):**
- Everything: $0.00 (completely free!)

**Recommendation:** 
- Start with RAG mode for demos and important queries
- Use search-only mode for exploration and testing

## Security Best Practices

1. **Never commit .env to git** (already in .gitignore)
2. **Rotate keys regularly** in Azure Portal
3. **Use separate keys** for dev/test/prod
4. **Monitor usage** in Azure Portal → Cost Management
5. **Set spending limits** in Azure to prevent surprises

## What Happens When You Configure Azure OpenAI?

**Without Azure OpenAI (Search-Only Mode):**
```
You: What does process X do?

Chatbot: [Returns top 5 search results with document snippets]
```

**With Azure OpenAI (RAG Mode):**
```
You: What does process X do?

Chatbot: Process X is a patient matching workflow that reads data 
from the patientacctspayercob table, performs deduplication using 
SSN and DOB fields, and outputs matched records to the 
matched_patients table. It's part of the CDD (Coverage Discovery 
and Deduplication) system and processes approximately 10,000 records 
per day...

[Includes sources and confidence score]
```

The AI generates natural language answers based on your codebase!
