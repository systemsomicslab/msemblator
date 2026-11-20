"""Per-tool formula annotation runners for the SIRIUS-free model.

Each runner writes, under <output>/<tool>/:
    <tool>_candidates.csv   query_id, tool, rank, formula, score, + tool-specific columns
    <tool>_status.csv       query_id, tool, status, n_candidates, message (one row per spectrum)
    manifest.json           tool/package versions, parameters, input checksum, timings
"""
