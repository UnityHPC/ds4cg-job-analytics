# Preprocess Data

## Preprocessing Criteria for Job Data
### Attributes Omitted
- **UUID**
- **Nodes**: NodesList have more specific information
- **Preempted**: Contains unreliable data. Use Status column instead (PREEMPT for
    unfinished, COMPLETE/FAILED/etc. for finished preempted jobs).
- **EndTime**: Can be calculated from StartTime and Elapsed

### Options for Including or Omitting Jobs
- **Keeping CPU jobs:**
    - If `GPUType` is null, the value will be filled with `["cpu"]`
    - If `GPUs` is null or is 0, the value will be 0.
- **Keeping jobs where the status is "Failed" or "Cancelled"**
- **Keeping jobs where the QOS is customized (not normal, long, or short)**

### Records Omitted If:
- `Elapsed` is less than the minimum threshold
- `account` is root
- `partition` is building
- `QOS` is updates

### Null Attribute Defaults
- `ArrayID`: set to -1
- `Interactive`: set to `"non-interactive"`
- `Constraints`: set to an empty numpy array
- `GPUs`: set to 0 (when CPU jobs are kept)
- `GPUType`: set to an numpy array ["cpu"] (when CPU jobs are kept)

### Attribute Types
- `StartTime`, `SubmitTime`: **datetime**
- `TimeLimit`, `Elapsed`: **timedelta**
- `Interactive`, `Status`, `ExitCode`, `QOS`, `Partition`, `Account`: **Categorical**

::: src.preprocess.preprocess