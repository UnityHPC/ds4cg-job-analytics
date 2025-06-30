# Preprocess Data

# Preprocessing Criteria for Job Data
## Attributes Omitted
- **UUID**
- **Nodes**: NodesList have more specific information
- **EndTime**: Can be calculated from StartTime and Elapsed

## Options for Including/Omitting Jobs
- **Keeping CPU jobs:**
    - If `GPUType` is null, the value will be filled with `["cpu"]`
    - If `GPUs` is null or is 0, the value will be 0.
- **Keeping jobs where the status is "Failed" or "Cancelled"**

## Records Omitted If:
- `Elapsed` is less than the minimum threshold
- `account` is root
- `partition` is building
- `QOS` is updates

## Null Attribute Defaults
- `ArrayID`: set to -1
- `Interactive`: set to `"non-interactive"`
- `GPUs`: set to 0 (when CPU jobs are kept)
- `GPUType`: set to an numpy array ["cpu"] (when CPU jobs are kept)

## Nullable Attributes
- `Constraints`: nullable for integrity in data analysis
- `requested_vram`: nullable for integrity in data analysis. Will be nulled when Constraints is null.

## Attribute Types
- `StartTime`, `SubmitTime`: **datetime**
- `TimeLimit`, `Elapsed`: **timedelta**
- `Interactive`, `Status`, `ExitCode`, `QOS`, `Partition`, `Account`: **Categorical**



::: src.preprocess.preprocess