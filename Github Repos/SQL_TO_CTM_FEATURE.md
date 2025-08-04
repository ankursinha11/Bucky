# SQL to CTM Mapping Feature

## Overview

The SQL to CTM (Control Table Mapping) feature allows users to upload SQL files and convert them into comprehensive CTM mapping documents using Bucky's collaborative mode. This feature integrates with the existing application-modernization tool and leverages Bucky's AI capabilities for intelligent SQL analysis.

## Features

### 1. SQL File Upload and Analysis
- Support for `.sql` and `.txt` file formats
- Automatic SQL syntax validation
- Structure analysis (tables, columns, joins, transformations)
- Performance analysis and optimization suggestions

### 2. CTM Mapping Generation
The system generates comprehensive CTM mapping documents with the following sections:

#### 🧾 SECTION 1: FEED METADATA
- **FEED NAME**: Extracted from table names or file context
- **OWNERS**: Extracted from comments or context
- **FILE VALIDATION SCENARIOS**:
  - COMBINE FILES (Yes/No based on UNION operations)
  - PARSE HEADER (Yes/No based on file operations)
  - MULTIPLE FILES (Yes/No based on multiple sources)
  - ADDITIONAL RECORDS (Yes/No based on INSERT/UPDATE)
  - PARSE TRAILER (Yes/No based on file operations)

#### 🧾 SECTION 2: RECORD VALIDATION
- Summarizes validation logic from WHERE clauses and filters
- Identifies business rules and constraints
- Documents data quality checks

#### 🧾 SECTION 3: TRANSFORMATION MAPPING TABLE
| TARGET TABLE | TARGET ATTRIBUTE | TARGET TYPE | SEQUENCE | RULE | SOURCE COLUMN | SOURCE TABLE | COMMENTS |

#### 🧾 JOIN SECTION
| JOIN TYPE | TABLE 1 | TABLE 2 | CONDITION |

### 3. Bucky Integration
- **Collaborative Mode**: Bucky works with users to analyze SQL and generate mappings
- **Interactive Analysis**: Users can ask questions and request clarifications
- **Intelligent Suggestions**: Bucky provides recommendations for mapping improvements

## Technical Implementation

### Backend Components

#### 1. SQL Analysis Agent (`SQLAnalysisAgent`)
```python
class SQLAnalysisAgent(SpecializedAgent):
    """Agent specialized in SQL analysis and CTM mapping generation"""
    
    Capabilities:
    - SQL Analysis: Analyze SQL code structure
    - CTM Mapping Generation: Generate CTM documents
    - SQL Validation: Validate syntax and identify issues
    - Join Analysis: Analyze JOIN operations
```

#### 2. Enhanced LLM Methods
```python
def sql_to_ctm_mapping(self, input_files: List[io.BytesIO]) -> Dict[str, Any]:
    """Convert SQL files to CTM format"""
```

#### 3. New API Endpoints
```python
@router.post("/sql-to-ctm", response_model=ConversionResponse)
async def sql_to_ctm_conversion(
    files: List[UploadFile] = File(...),
    user_id: str = Form(...),
    mode: str = Form("collaborative")
):
```

### Frontend Components

#### 1. Updated Language Support
```typescript
const supportedLanguages = {
  source: [".NET", "COBOL", "C++", "ETL", "Delphi", "C", "DataStage", "SQL"],
  target: ["PySpark", "PySpark and DBT", "Java", "CTM"],
}
```

#### 2. New Service Method
```typescript
static async sqlToCTM(
  files: File[],
  userId: string,
  mode: string = "collaborative"
): Promise<ConversionResponse>
```

## Usage Instructions

### 1. Upload SQL Files
1. Navigate to the Application Modernization tool
2. Select "SQL" as the source language
3. Select "CTM" as the target language
4. Upload your SQL files (`.sql` or `.txt` format)

### 2. Generate CTM Mapping
1. Click "Generate Mapping" button
2. Bucky will analyze your SQL in collaborative mode
3. Review the generated CTM mapping document
4. Ask Bucky for clarifications or modifications as needed

### 3. Collaborative Interaction
- Ask Bucky questions about the mapping
- Request modifications to specific sections
- Get explanations for complex transformations
- Request additional documentation

## Example SQL to CTM Conversion

### Input SQL:
```sql
SELECT 
    c.customer_id,
    c.customer_name,
    CASE 
        WHEN o.total_amount > 1000 THEN 'High Value'
        ELSE 'Standard'
    END as customer_segment
FROM customers c
INNER JOIN orders o ON c.customer_id = o.customer_id
WHERE o.order_date >= '2023-01-01'
```

### Generated CTM Mapping:
```
🧾 SECTION 1: FEED METADATA
- FEED NAME: customer_order_analysis
- OWNERS: data_team

FILE VALIDATION SCENARIOS:
- COMBINE FILES: No
- PARSE HEADER: No
- MULTIPLE FILES: No
- ADDITIONAL RECORDS: No
- PARSE TRAILER: No

🧾 SECTION 2: RECORD VALIDATION
- Filter orders from 2023-01-01 onwards
- Only include completed orders
- Validate customer_id exists in both tables

🧾 SECTION 3: TRANSFORMATION MAPPING TABLE
| TARGET TABLE | TARGET ATTRIBUTE | TARGET TYPE | SEQUENCE | RULE | SOURCE COLUMN | SOURCE TABLE | COMMENTS |
|--------------|------------------|--------------|----------|------|----------------|---------------|----------|
| customer_analysis | customer_id | VARCHAR | 1 | Direct mapping | customer_id | customers | Primary key |
| customer_analysis | customer_name | VARCHAR | 2 | Direct mapping | customer_name | customers | Customer name |
| customer_analysis | customer_segment | VARCHAR | 3 | CASE WHEN total_amount > 1000 THEN 'High Value' ELSE 'Standard' END | total_amount | orders | Business logic |

🧾 JOIN SECTION
| JOIN TYPE | TABLE 1 | TABLE 2 | CONDITION |
|-----------|---------|---------|-----------|
| INNER JOIN | customers | orders | customer_id = customer_id |
```

## Error Handling

### 1. Mapping Validation Fix
- **Issue**: `mapping_doc` field expected dictionary but received list
- **Fix**: Updated `generate_mapping_records` method to always return dictionary format
- **Result**: Eliminates Pydantic validation errors

### 2. SQL Analysis Errors
- **Handling**: Comprehensive error handling in SQL analysis agent
- **Fallback**: Graceful degradation with error messages
- **User Feedback**: Clear error messages with suggestions

## Testing

### Test Files
- `test_sql_sample.sql`: Sample SQL file for testing
- Contains various SQL constructs (JOINs, CASE statements, aggregations)

### Test Scenarios
1. **Basic SQL Analysis**: Simple SELECT statements
2. **Complex Transformations**: CASE statements, window functions
3. **Multiple JOINs**: Complex join scenarios
4. **Error Handling**: Invalid SQL syntax
5. **Collaborative Mode**: User interaction scenarios

## Future Enhancements

### 1. Advanced SQL Features
- Support for stored procedures
- Complex subquery analysis
- Dynamic SQL parsing
- Performance optimization suggestions

### 2. Enhanced CTM Mapping
- Custom mapping templates
- Industry-specific mappings
- Automated validation rules
- Integration with existing CTM tools

### 3. Bucky Integration
- Voice interaction for SQL analysis
- Real-time collaboration features
- Advanced query optimization suggestions
- Integration with data lineage tools

## Troubleshooting

### Common Issues

1. **SQL File Not Recognized**
   - Ensure file has `.sql` or `.txt` extension
   - Check file encoding (UTF-8 recommended)
   - Verify SQL syntax is valid

2. **CTM Mapping Incomplete**
   - Review SQL for complex transformations
   - Ask Bucky for clarification on unclear mappings
   - Check for missing table or column information

3. **Bucky Not Responding**
   - Check network connectivity
   - Verify Bucky service is running
   - Try refreshing the page

### Support
For issues or questions, contact the development team or check the application logs for detailed error information. 