"""Recognize deliberately pruned Grace results without weakening new runs.

Only the immutable migration inventory can authorize a historical result whose
status or raw tensor was removed. Newly generated uniform results continue to
use the original study's complete() predicate, including every saved artifact.
"""
import hashlib
import json
from pathlib import Path


ROOT = Path('/home/shig/diffpano')
SUITE = ROOT / 'outputs/10.15layout24-flux216/seed0-v1'
INVENTORY = Path(
    '/scratch/user/shig/diffpano/migration-to-faster-20261016/'
    'pending-panorama-slots.json'
)
INVENTORY_SHA256 = 'f3c500078f6e4344faf924fc323151cdcde9470ab2f20581b3c3e2b21e8f21aa'
MANIFEST_SHA256 = '3f71e8d02c0254759703fb102fa77704c4eff5cd72b11fdf31437acdbcbb560d'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    with Path(path).open() as stream:
        return json.load(stream)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


class CompletionVerifier:
    """Verifier built from already authenticated manifest/inventory evidence.

    Production callers should use load_verifier(), which authenticates both
    files against pinned hashes. The constructor is separate for isolated tests.
    """

    def __init__(self, manifest, inventory, suite_root, strict_complete):
        self.suite_root = Path(suite_root)
        self.strict_complete = strict_complete
        rows = manifest['rows']
        self.rows = {row['index']: row for row in rows}
        _require(len(self.rows) == len(rows), 'Duplicate manifest indices')
        _require(len({r['case_id'] for r in rows}) == len(rows),
                 'Duplicate manifest case identities')
        protected = inventory['completed_rows']
        self.protected_records = {r['index']: r for r in protected}
        _require(len(self.protected_records) == len(protected),
                 'Duplicate completed indices')
        pending = inventory['pending_rows']
        pending_by_index = {r['index']: r for r in pending}
        _require(len(pending_by_index) == len(pending), 'Duplicate pending indices')
        _require(not set(pending_by_index) & set(self.protected_records),
                 'Completed and pending inventories overlap')
        _require(set(pending_by_index) | set(self.protected_records) == set(self.rows),
                 'Migration inventory does not partition the manifest')
        for record in protected + pending:
            fixed = self.rows[record['index']]
            for key in ('case_id', 'semantic_sha256', 'source_sha256',
                        'variant', 'prompt', 'layout', 'camera_count', 'subset'):
                _require(record[key] == fixed[key],
                         'Migration identity differs from manifest: ' + key)
            relative = Path(fixed['case_id'])
            _require(not relative.is_absolute() and '..' not in relative.parts,
                     'Invalid case path')
            expected_image = self.suite_root / 'cases' / relative / 'final.png'
            recorded_path = record.get('image_path', record.get('expected_image_path'))
            _require(recorded_path == str(expected_image),
                     'Migration image path differs from suite case')
        self.fresh_indices = frozenset(
            index for index, row in pending_by_index.items()
            if row['layout'].startswith('uniform')
        )

    def _fixed_row(self, row):
        """Allow evaluator snapshot fields, never altered manifest fields."""
        try:
            fixed = self.rows[row['index']]
            return fixed if all(row.get(k) == v for k, v in fixed.items()) else None
        except (KeyError, TypeError):
            return None

    def protected_complete(self, row):
        """Verify a named legacy result using its immutable retained artifacts."""
        fixed = self._fixed_row(row)
        if fixed is None or fixed['index'] not in self.protected_records:
            return False
        record = self.protected_records[fixed['index']]
        directory = self.suite_root / 'cases' / fixed['case_id']
        try:
            if sha256(directory / 'final.png') != record['image_sha256_recorded']:
                return False
            if sha256(directory / 'metadata.json') != record['metadata_sha256_recorded']:
                return False
            metadata = read_json(directory / 'metadata.json')
            if metadata['artifacts']['final.png'] != record['image_sha256_recorded']:
                return False
            if metadata['reused'] != record['reused']:
                return False
            if metadata['reused']:
                # Historical sources have their own source revision. Their
                # authenticated suite wrapper supplies the logical case identity.
                return all(metadata[key] == fixed[key]
                           for key in ('case_id', 'prompt', 'variant'))
            expected_case = {k: v for k, v in fixed.items() if k != 'semantic'}
            return (
                metadata['case'] == expected_case
                and metadata['semantic_sha256'] == fixed['semantic_sha256']
                and metadata['source_sha256'] == fixed['source_sha256']
            )
        except (OSError, ValueError, KeyError, TypeError):
            return False

    def checked_complete(self, row, verify=True):
        """Use pruned evidence only for the 101 known cases; strict for new 75."""
        fixed = self._fixed_row(row)
        if fixed is None:
            return False
        if fixed['index'] in self.protected_records:
            # The caller cannot request weaker legacy checks through verify=False.
            return self.protected_complete(fixed)
        if fixed['index'] not in self.fresh_indices:
            return False
        return self.strict_complete(fixed, verify=verify)


def load_verifier(strict_complete=None):
    """Load pinned migration evidence and preserve the original fresh-run check."""
    _require(sha256(INVENTORY) == INVENTORY_SHA256, 'Migration inventory hash changed')
    _require(sha256(SUITE / 'manifest.json') == MANIFEST_SHA256,
             'Frozen panorama manifest hash changed')
    inventory = read_json(INVENTORY)
    manifest = read_json(SUITE / 'manifest.json')
    _require(inventory['manifest']['sha256'] == MANIFEST_SHA256,
             'Migration manifest reference differs')
    _require(inventory['completed_count'] == 101 and inventory['pending_count'] == 115,
             'Unexpected migration inventory counts')
    if strict_complete is None:
        from studies.layout24.common import complete
        strict_complete = complete
    verifier = CompletionVerifier(manifest, inventory, SUITE, strict_complete)
    _require(len(verifier.rows) == 216 and len(verifier.protected_records) == 101,
             'Unexpected fixed study size')
    _require(len(verifier.fresh_indices) == 75, 'Unexpected uniform resume scope')
    return verifier
