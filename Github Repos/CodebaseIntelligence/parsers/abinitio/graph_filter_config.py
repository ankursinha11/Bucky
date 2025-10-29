"""
Ab Initio Graph Filter Configuration
======================================
Defines which graphs to parse based on business modules

Only these 36 critical graphs will be parsed and exported.
"""

# List of graph names to include (without .mp/.plan extension)
INCLUDED_GRAPHS = [
    # Commercial Generation (11 graphs)
    "100_commGenPrePrep",
    "105_commGenPrePrep",
    "400_commGenIpa",
    "405_commGenPatcho",
    "410_commGenPrePA",
    "415_commGenResultGmrn",
    "420_commGenFinal",
    "430_commGenFinalCluster",
    "435_commGenClusteringReport",
    "500_commGenLoadFinalFile",
    "505_GenLoadFinalFile",

    # Medicare Leads Generation (6 graphs)
    "120_mcarePrePrep",
    "440_mCareGenIpa",
    "445_mCareGenDsh",
    "450_mCareGenHets",
    "455_mCareGenHetsOnly",
    "460_mCareGenFinal",

    # CDD (15 graphs)
    "1000_CDD_PrePrep",
    "1100_CDD_Charlotte271Data",
    "1200_CDD_Charlotte271FamilyData",
    "1300_CDD_PatientAcctsXRefPermID",
    "1400_CDD_Charlotte271MRNData",
    "1500_CDD_TUSourcedFamilyMemberLink",
    "1600_CDD_HFC_FamilyFoundCoverage",
    "1700_CDD_HFC_RelatedMemberOPSourcedFC",
    "1800_CDD_HFC_Charlotte271Data",
    "2000_CDD_LoadStagingAndCallISP",
    "2200_CDD_LoadHelperFoundCoveragesAndCallISP",
    "2500_CDD_PropagateHFCForFamilyMembers",
    "2800_CDD_LoadHelperFoundCoveragesAndCallISP_Propagation",

    # GHIC (1 graph)
    "439_LoadSnavGlobalMRNXHospInsuranceCodes",

    # Data Ingestion (5 graphs - these are .plan files)
    "200_extractDataFromSqlToAbi",
    "210_compareDataInAbiToSql",
    "265_fileTransferToHadoopServer",
    "300_extractDataFromSqlToAbi_FasterETL",
    "600_consolidateArchiveCleanUp",
]

# Module mapping for organization
MODULE_MAP = {
    "Commercial Generation": [
        "100_commGenPrePrep", "105_commGenPrePrep", "400_commGenIpa",
        "405_commGenPatcho", "410_commGenPrePA", "415_commGenResultGmrn",
        "420_commGenFinal", "430_commGenFinalCluster", "435_commGenClusteringReport",
        "500_commGenLoadFinalFile", "505_GenLoadFinalFile"
    ],
    "Medicare Leads Generation": [
        "120_mcarePrePrep", "440_mCareGenIpa", "445_mCareGenDsh",
        "450_mCareGenHets", "455_mCareGenHetsOnly", "460_mCareGenFinal"
    ],
    "CDD": [
        "1000_CDD_PrePrep", "1100_CDD_Charlotte271Data", "1200_CDD_Charlotte271FamilyData",
        "1300_CDD_PatientAcctsXRefPermID", "1400_CDD_Charlotte271MRNData",
        "1500_CDD_TUSourcedFamilyMemberLink", "1600_CDD_HFC_FamilyFoundCoverage",
        "1700_CDD_HFC_RelatedMemberOPSourcedFC", "1800_CDD_HFC_Charlotte271Data",
        "2000_CDD_LoadStagingAndCallISP", "2200_CDD_LoadHelperFoundCoveragesAndCallISP",
        "2500_CDD_PropagateHFCForFamilyMembers", "2800_CDD_LoadHelperFoundCoveragesAndCallISP_Propagation"
    ],
    "GHIC": ["439_LoadSnavGlobalMRNXHospInsuranceCodes"],
    "Data Ingestion": [
        "200_extractDataFromSqlToAbi", "210_compareDataInAbiToSql",
        "265_fileTransferToHadoopServer", "300_extractDataFromSqlToAbi_FasterETL",
        "600_consolidateArchiveCleanUp"
    ]
}


def is_graph_included(file_name: str) -> bool:
    """
    Check if a graph should be included based on filter

    Args:
        file_name: Graph file name (with or without extension)

    Returns:
        True if graph should be included
    """
    # Remove extension if present
    base_name = file_name.replace(".mp", "").replace(".plan", "").replace(".pset", "")

    # Check if in included list
    return base_name in INCLUDED_GRAPHS


def get_module_for_graph(graph_name: str) -> str:
    """
    Get the module name for a graph

    Args:
        graph_name: Graph name

    Returns:
        Module name or "Unknown"
    """
    base_name = graph_name.replace(".mp", "").replace(".plan", "").replace(".pset", "")

    for module, graphs in MODULE_MAP.items():
        if base_name in graphs:
            return module

    return "Unknown"
