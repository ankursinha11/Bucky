"""
Ab Initio Parser Patterns
Pattern definitions for parsing .mp files
"""

# Component types to extract
ALLOWED_COMPONENT_TYPES = [
    "Input_File",
    "Output_File",
    "Lookup_File",
    "Transform",
    "Reformat",
    "Join",
    "Filter",
    "Aggregate",
    "Rollup",
    "Sort",
    "Dedup",
    "Gather",
    "Partition",
    "Scan",
    "Run_Program",
]

# Patterns dictionary
PATTERNS = {
    "ALLOWED_COMPONENT_TYPES": ALLOWED_COMPONENT_TYPES,
    "PARAMETER_SECTION_PATTERN": r'XXparameter_set\|@@@@\{\{(.*?)\}\}@',
    "PARAMETER_PATTERN": r'\{(\d+)\|XXparameter\|!?([^|]+)\|([^|]*)\|([^|]*)\|([^|]*)\|([^|]*)\|([^}]*)\}',
    "COMPONENT_PATTERN": r'\{(\d+)\|([^|]+)\|([^|]+)\|',
    "GRAPH_ID": "30001001",
    "BLOCK_INTERNAL_PATTERN": r'\{@@@@',
}


def get_patterns() -> dict:
    """
    Get patterns dictionary

    Returns:
        Dictionary of patterns
    """
    return PATTERNS
