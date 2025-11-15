# MongoDB GridFS Model Storage

## Overview

Trained ML models are now stored in **MongoDB GridFS** for persistent storage across deployments. This ensures models are not lost when containers restart or are redeployed.

## How It Works

### Storage Strategy: Hybrid Approach

1. **Primary Storage**: MongoDB GridFS (persistent)
   - Models stored in `models` collection
   - Metadata stored in `model_metadata` collection
   - Survives container restarts and deployments

2. **Local Cache**: File system (performance)
   - Models cached locally in `models/` directory
   - Faster access for predictions
   - Auto-synced from GridFS on first load

### Storage Flow

**When Training:**
```
Train Model → Save to GridFS → Save to Local Cache → Update Metadata
```

**When Loading:**
```
Check Local Cache → If exists, load from cache
                   → If not, load from GridFS → Save to cache → Load model
```

## Benefits

✅ **Persistent**: Models survive deployments  
✅ **Fast**: Local cache for quick access  
✅ **Automatic**: No manual sync needed  
✅ **Backward Compatible**: Works without MongoDB (falls back to local only)  
✅ **No Extra Cost**: Uses existing MongoDB infrastructure  

## MongoDB Collections

### `models` Collection (GridFS)
Stores the actual model files:
- Filename format: `{model_name}/{version}.joblib`
- Example: `transaction_categorizer/latest.joblib`
- Includes metadata: model_name, version, upload_date

### `model_metadata` Collection
Stores model metadata:
- Document ID: `"model_metadata"`
- Contains: version info, training dates, categories, etc.

## Configuration

GridFS is **automatically enabled** when:
- MongoDB database connection is available
- `gridfs` module is available (included with pymongo)

No additional configuration needed! The system automatically:
- Detects MongoDB connection
- Enables GridFS storage
- Falls back to local-only if MongoDB unavailable

## Migration

### Existing Models

If you have existing models in `models/` directory:
1. They will continue to work (local cache)
2. Next time you train, the new model will be saved to GridFS
3. Old models can be manually migrated by retraining

### New Deployments

On a fresh deployment:
1. Models are automatically loaded from GridFS
2. Cached locally for performance
3. No manual intervention needed

## Verification

Check if GridFS is working:

```python
# Check model info
GET /api/v1/transactions/model-info

# Response will show:
{
  "stored_in_gridfs": true,  # ← Indicates GridFS storage
  "gridfs_filename": "transaction_categorizer/latest.joblib"
}
```

## Troubleshooting

### Models not persisting?
- Check MongoDB connection is working
- Verify `gridfs` is available (comes with pymongo)
- Check MongoDB logs for errors

### Fallback behavior
- If GridFS fails, models save to local cache only
- System continues to work (just not persistent)
- Check logs for GridFS errors

## File Locations

**Local Cache:**
- `models/transaction_categorizer_latest.joblib`
- `models/model_metadata.json`

**MongoDB GridFS:**
- Collection: `models.files` and `models.chunks`
- Metadata: `model_metadata` collection

## Best Practices

1. **Train regularly**: Models are automatically saved to GridFS
2. **Monitor storage**: Check MongoDB storage usage
3. **Backup MongoDB**: GridFS files are included in MongoDB backups
4. **Version models**: Use version strings for model tracking

## Example: Training with GridFS

```bash
# Train a model (automatically saves to GridFS)
POST /api/v1/transactions/train-model
# File: training_data.csv

# Model is now:
# ✅ Stored in MongoDB GridFS
# ✅ Cached locally
# ✅ Available after deployment
```

## Technical Details

- **GridFS Collection**: `models` (default)
- **Metadata Collection**: `model_metadata`
- **File Format**: `.joblib` (scikit-learn models)
- **Max File Size**: MongoDB GridFS supports files up to 16MB per chunk (handles larger files automatically)

