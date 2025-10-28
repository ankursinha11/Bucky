"""
Process Matcher
Matches processes across different systems (Ab Initio, Hadoop, Databricks)
"""

from typing import List, Dict, Any, Tuple
from loguru import logger

from core.models import Process, Component


class ProcessMatcher:
    """Match processes across systems"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.threshold = self.config.get("threshold", 0.7)

    def match_processes(
        self,
        source_processes: List[Process],
        target_processes: List[Process],
        source_components: List[Component],
        target_components: List[Component],
    ) -> Dict[str, Tuple[str, float]]:
        """
        Match source processes to target processes
        Returns: {source_process_id: (target_process_id, confidence)}
        """
        logger.info(
            f"Matching {len(source_processes)} source processes to {len(target_processes)} target processes"
        )

        matches = {}

        for source_proc in source_processes:
            best_match = None
            best_score = 0.0

            for target_proc in target_processes:
                score = self._calculate_similarity(
                    source_proc, target_proc, source_components, target_components
                )

                if score > best_score and score >= self.threshold:
                    best_score = score
                    best_match = target_proc.id

            if best_match:
                matches[source_proc.id] = (best_match, best_score)
                logger.info(
                    f"Matched {source_proc.name} -> {next(p.name for p in target_processes if p.id == best_match)} (score: {best_score:.2f})"
                )

        logger.info(f"Matched {len(matches)} out of {len(source_processes)} processes")
        return matches

    def _calculate_similarity(
        self,
        source_proc: Process,
        target_proc: Process,
        source_comps: List[Component],
        target_comps: List[Component],
    ) -> float:
        """Calculate similarity score between two processes"""
        scores = []
        weights = []

        # 1. Name similarity (weight: 0.2)
        name_score = self._name_similarity(source_proc.name, target_proc.name)
        scores.append(name_score)
        weights.append(0.2)

        # 2. Table/dataset overlap (weight: 0.4)
        table_score = self._table_overlap(source_proc, target_proc)
        scores.append(table_score)
        weights.append(0.4)

        # 3. Business function similarity (weight: 0.2)
        business_score = self._business_function_similarity(source_proc, target_proc)
        scores.append(business_score)
        weights.append(0.2)

        # 4. Component similarity (weight: 0.2)
        source_proc_comps = [c for c in source_comps if c.process_id == source_proc.id]
        target_proc_comps = [c for c in target_comps if c.process_id == target_proc.id]
        component_score = self._component_similarity(source_proc_comps, target_proc_comps)
        scores.append(component_score)
        weights.append(0.2)

        # Weighted average
        total_score = sum(s * w for s, w in zip(scores, weights))

        return total_score

    def _name_similarity(self, name1: str, name2: str) -> float:
        """Calculate name similarity using string matching"""
        name1_lower = name1.lower()
        name2_lower = name2.lower()

        # Exact match
        if name1_lower == name2_lower:
            return 1.0

        # Substring match
        if name1_lower in name2_lower or name2_lower in name1_lower:
            return 0.8

        # Token overlap
        tokens1 = set(name1_lower.split("_"))
        tokens2 = set(name2_lower.split("_"))
        common_tokens = tokens1 & tokens2

        if not tokens1 or not tokens2:
            return 0.0

        jaccard = len(common_tokens) / len(tokens1 | tokens2)
        return jaccard

    def _table_overlap(self, proc1: Process, proc2: Process) -> float:
        """Calculate table/dataset overlap"""
        tables1 = set(proc1.tables_involved + proc1.input_sources + proc1.output_targets)
        tables2 = set(proc2.tables_involved + proc2.input_sources + proc2.output_targets)

        # Clean table names (remove paths, extensions)
        tables1_clean = {self._clean_table_name(t) for t in tables1}
        tables2_clean = {self._clean_table_name(t) for t in tables2}

        if not tables1_clean or not tables2_clean:
            return 0.0

        # Jaccard similarity
        intersection = tables1_clean & tables2_clean
        union = tables1_clean | tables2_clean

        return len(intersection) / len(union)

    def _clean_table_name(self, table: str) -> str:
        """Extract clean table name from path or full name"""
        # Remove paths
        if "/" in table:
            table = table.split("/")[-1]

        # Remove extensions
        if "." in table:
            table = table.split(".")[0]

        return table.lower()

    def _business_function_similarity(self, proc1: Process, proc2: Process) -> float:
        """Compare business functions"""
        func1 = (proc1.business_function or proc1.description or "").lower()
        func2 = (proc2.business_function or proc2.description or "").lower()

        if not func1 or not func2:
            return 0.0

        # Check for common keywords
        keywords = [
            "lead",
            "coverage",
            "gmrn",
            "patient",
            "cdd",
            "edi",
            "demographics",
            "matching",
        ]

        func1_keywords = {kw for kw in keywords if kw in func1}
        func2_keywords = {kw for kw in keywords if kw in func2}

        if not func1_keywords and not func2_keywords:
            return 0.5  # Neutral

        if not func1_keywords or not func2_keywords:
            return 0.0

        jaccard = len(func1_keywords & func2_keywords) / len(func1_keywords | func2_keywords)
        return jaccard

    def _component_similarity(
        self, comps1: List[Component], comps2: List[Component]
    ) -> float:
        """Compare component types and counts"""
        if not comps1 or not comps2:
            return 0.0

        # Compare component type distribution
        types1 = [c.component_type for c in comps1]
        types2 = [c.component_type for c in comps2]

        # Count occurrences
        from collections import Counter

        count1 = Counter(types1)
        count2 = Counter(types2)

        # Calculate similarity
        all_types = set(count1.keys()) | set(count2.keys())
        if not all_types:
            return 0.0

        similarity = 0.0
        for comp_type in all_types:
            c1 = count1.get(comp_type, 0)
            c2 = count2.get(comp_type, 0)
            similarity += 1 - abs(c1 - c2) / max(c1, c2, 1)

        return similarity / len(all_types)


from typing import Optional  # Add missing import
