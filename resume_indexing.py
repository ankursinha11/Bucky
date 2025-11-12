"""
Resume/Incremental Indexing System
Allows indexing to resume from where it left off after crashes or restarts
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Optional
from dataclasses import dataclass, asdict
from loguru import logger


@dataclass
class IndexingCheckpoint:
    """Represents a checkpoint in the indexing process"""
    system_type: str
    repository_path: str
    total_files: int
    files_processed: int
    files_indexed: int
    files_failed: int
    last_processed_file: str
    processed_file_hashes: Set[str]
    failed_files: List[str]
    timestamp: str
    checkpoint_id: str

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            **asdict(self),
            'processed_file_hashes': list(self.processed_file_hashes)  # Convert set to list
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'IndexingCheckpoint':
        """Create from dictionary"""
        data['processed_file_hashes'] = set(data.get('processed_file_hashes', []))
        return cls(**data)


class ResumeIndexingManager:
    """
    Manages resume/incremental indexing with checkpoints

    Features:
    - Saves progress after every N files
    - Can resume from last checkpoint
    - Tracks failed files for retry
    - Detects file changes (via hash)
    - Supports incremental updates
    """

    def __init__(self, checkpoint_dir: str = "./outputs/checkpoints"):
        """Initialize resume indexing manager"""
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_frequency = 10  # Save checkpoint every 10 files
        self.current_checkpoint: Optional[IndexingCheckpoint] = None

    def start_indexing(
        self,
        system_type: str,
        repository_path: str,
        files_to_process: List[Path],
        force_reindex: bool = False
    ) -> tuple[List[Path], Optional[IndexingCheckpoint]]:
        """
        Start or resume indexing

        Returns:
            (files_to_process, previous_checkpoint)
        """
        checkpoint_path = self._get_checkpoint_path(system_type, repository_path)

        # Try to load existing checkpoint
        previous_checkpoint = None
        if not force_reindex and checkpoint_path.exists():
            try:
                previous_checkpoint = self.load_checkpoint(checkpoint_path)
                logger.info(f"📂 Found checkpoint from {previous_checkpoint.timestamp}")
                logger.info(f"   Previously processed: {previous_checkpoint.files_processed}/{previous_checkpoint.total_files}")

                # Filter out already processed files
                files_to_process = self._filter_processed_files(
                    files_to_process,
                    previous_checkpoint
                )

                logger.info(f"   Remaining files: {len(files_to_process)}")

            except Exception as e:
                logger.warning(f"Could not load checkpoint: {e}")
                previous_checkpoint = None

        # Create new checkpoint
        checkpoint_id = self._generate_checkpoint_id(system_type, repository_path)

        self.current_checkpoint = IndexingCheckpoint(
            system_type=system_type,
            repository_path=repository_path,
            total_files=len(files_to_process) + (previous_checkpoint.files_processed if previous_checkpoint else 0),
            files_processed=previous_checkpoint.files_processed if previous_checkpoint else 0,
            files_indexed=previous_checkpoint.files_indexed if previous_checkpoint else 0,
            files_failed=previous_checkpoint.files_failed if previous_checkpoint else 0,
            last_processed_file="",
            processed_file_hashes=previous_checkpoint.processed_file_hashes if previous_checkpoint else set(),
            failed_files=previous_checkpoint.failed_files if previous_checkpoint else [],
            timestamp=datetime.now().isoformat(),
            checkpoint_id=checkpoint_id
        )

        return files_to_process, previous_checkpoint

    def mark_file_processed(
        self,
        file_path: Path,
        success: bool = True,
        auto_save: bool = True
    ):
        """
        Mark a file as processed

        Args:
            file_path: Path to the processed file
            success: Whether processing succeeded
            auto_save: Whether to auto-save checkpoint based on frequency
        """
        if not self.current_checkpoint:
            return

        # Calculate file hash
        file_hash = self._calculate_file_hash(file_path)

        # Update checkpoint
        self.current_checkpoint.files_processed += 1
        self.current_checkpoint.last_processed_file = str(file_path)
        self.current_checkpoint.processed_file_hashes.add(file_hash)

        if success:
            self.current_checkpoint.files_indexed += 1
        else:
            self.current_checkpoint.files_failed += 1
            self.current_checkpoint.failed_files.append(str(file_path))

        # Auto-save based on frequency
        if auto_save and (self.current_checkpoint.files_processed % self.checkpoint_frequency == 0):
            self.save_checkpoint()

    def save_checkpoint(self) -> Path:
        """Save current checkpoint to disk"""
        if not self.current_checkpoint:
            logger.warning("No checkpoint to save")
            return None

        checkpoint_path = self._get_checkpoint_path(
            self.current_checkpoint.system_type,
            self.current_checkpoint.repository_path
        )

        try:
            with open(checkpoint_path, 'w') as f:
                json.dump(self.current_checkpoint.to_dict(), f, indent=2)

            logger.debug(f"💾 Checkpoint saved: {self.current_checkpoint.files_processed}/{self.current_checkpoint.total_files}")
            return checkpoint_path

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return None

    def load_checkpoint(self, checkpoint_path: Path) -> IndexingCheckpoint:
        """Load checkpoint from disk"""
        with open(checkpoint_path, 'r') as f:
            data = json.load(f)

        return IndexingCheckpoint.from_dict(data)

    def finalize_indexing(self, success: bool = True):
        """
        Finalize indexing - save final checkpoint and optionally delete it

        Args:
            success: If True, delete checkpoint (indexing complete)
                    If False, keep checkpoint for retry
        """
        if not self.current_checkpoint:
            return

        # Save final state
        checkpoint_path = self.save_checkpoint()

        if success:
            logger.info(f"✅ Indexing complete: {self.current_checkpoint.files_indexed} files indexed")
            logger.info(f"   Failed: {self.current_checkpoint.files_failed}")

            # Delete checkpoint (no need to resume)
            if checkpoint_path and checkpoint_path.exists():
                checkpoint_path.unlink()
                logger.info("   Checkpoint deleted (indexing complete)")

        else:
            logger.warning(f"⚠️ Indexing incomplete - checkpoint saved for resume")
            logger.info(f"   To resume, run indexing again (will auto-detect checkpoint)")

    def get_failed_files(self) -> List[str]:
        """Get list of files that failed processing"""
        if not self.current_checkpoint:
            return []
        return self.current_checkpoint.failed_files.copy()

    def retry_failed_files(self) -> List[Path]:
        """Get failed files for retry"""
        failed_files = self.get_failed_files()
        return [Path(f) for f in failed_files if Path(f).exists()]

    def _filter_processed_files(
        self,
        files: List[Path],
        checkpoint: IndexingCheckpoint
    ) -> List[Path]:
        """Filter out files that were already processed"""
        remaining_files = []

        for file_path in files:
            file_hash = self._calculate_file_hash(file_path)

            # Skip if already processed (and file hasn't changed)
            if file_hash in checkpoint.processed_file_hashes:
                continue

            remaining_files.append(file_path)

        return remaining_files

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate hash of file path and metadata (not content - too expensive)"""
        # Hash combination of path + size + mtime (faster than content hash)
        try:
            stat = file_path.stat()
            hash_input = f"{file_path}_{stat.st_size}_{stat.st_mtime}"
            return hashlib.md5(hash_input.encode()).hexdigest()
        except:
            # If stat fails, just hash the path
            return hashlib.md5(str(file_path).encode()).hexdigest()

    def _generate_checkpoint_id(self, system_type: str, repository_path: str) -> str:
        """Generate unique checkpoint ID"""
        hash_input = f"{system_type}_{repository_path}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]

    def _get_checkpoint_path(self, system_type: str, repository_path: str) -> Path:
        """Get checkpoint file path for a system"""
        checkpoint_id = self._generate_checkpoint_id(system_type, repository_path)
        return self.checkpoint_dir / f"checkpoint_{system_type}_{checkpoint_id}.json"

    def list_checkpoints(self) -> List[Dict]:
        """List all saved checkpoints"""
        checkpoints = []

        for checkpoint_file in self.checkpoint_dir.glob("checkpoint_*.json"):
            try:
                checkpoint = self.load_checkpoint(checkpoint_file)
                checkpoints.append({
                    "file": checkpoint_file.name,
                    "system": checkpoint.system_type,
                    "progress": f"{checkpoint.files_processed}/{checkpoint.total_files}",
                    "timestamp": checkpoint.timestamp,
                    "failed": checkpoint.files_failed
                })
            except:
                pass

        return checkpoints

    def clear_checkpoint(self, system_type: str, repository_path: str):
        """Clear checkpoint for a specific system (force fresh indexing)"""
        checkpoint_path = self._get_checkpoint_path(system_type, repository_path)
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            logger.info(f"🗑️  Cleared checkpoint for {system_type}")


# Example usage
def test_resume_indexing():
    """Test resume indexing functionality"""
    from pathlib import Path

    print("=" * 60)
    print("RESUME INDEXING TEST")
    print("=" * 60)

    # Simulate files to index
    test_repo = "/Users/ankurshome/Desktop/Hadoop_Parser/CodebaseIntelligence/hadoop_repos/hadoop_repos"
    files = list(Path(test_repo).rglob("*.pig"))[:20]  # First 20 Pig files

    print(f"\n📁 Found {len(files)} files to process")

    # Initialize manager
    manager = ResumeIndexingManager()

    # Start indexing
    files_to_process, previous_checkpoint = manager.start_indexing(
        system_type="hadoop",
        repository_path=test_repo,
        files_to_process=files,
        force_reindex=False
    )

    print(f"\n🚀 Starting indexing...")
    print(f"   Files to process: {len(files_to_process)}")

    # Simulate processing files
    for i, file_path in enumerate(files_to_process):
        print(f"   Processing {i+1}/{len(files_to_process)}: {file_path.name}")

        # Simulate processing
        success = True  # In real case, this depends on actual indexing

        # Mark as processed
        manager.mark_file_processed(file_path, success=success)

        # Simulate crash after 5 files
        if i == 4:
            print(f"\n💥 Simulating crash...")
            manager.save_checkpoint()
            break

    print(f"\n📊 Progress saved: {manager.current_checkpoint.files_processed} files")

    # List checkpoints
    print(f"\n📋 Saved checkpoints:")
    for cp in manager.list_checkpoints():
        print(f"   - {cp['system']}: {cp['progress']} (failed: {cp['failed']})")

    print("\n✓ Test complete - checkpoint saved for resume")
    print("=" * 60)


if __name__ == "__main__":
    test_resume_indexing()
