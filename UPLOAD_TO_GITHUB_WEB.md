# Upload with the GitHub website

This WEB-SAFE package contains no file larger than 9 MiB. It is intended for GitHub browser upload.

1. Do **not** upload the outer ZIP file.
2. Extract the ZIP locally.
3. Open the extracted folder `us365k-biomedclip-reproducibility-WEB-SAFE`.
4. Upload the contents to the repository root. If desired, upload in several batches by top-level folder.
5. Commit the upload.

Large artifacts were byte-split into `.part001`, `.part002`, ... files so they fit the browser uploader. The split map is in `reviewer_evidence/SPLIT_FILE_MANIFEST.json`.

To reconstruct them after cloning the repository:

```bash
python tools/recombine_large_files.py
```

The script verifies the SHA256 of every reconstructed original file.
