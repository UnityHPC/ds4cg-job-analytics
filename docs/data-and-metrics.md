# Data and Efficiency Metrics

This page provides comprehensive documentation about the data structure and efficiency metrics available in the DS4CG Unity Job Analytics project.

## Data Structure

The project works with job data from the Unity cluster's Slurm scheduler. After preprocessing, the data contains the following key attributes:

### Job Identification
- **JobID** – Unique identifier for each job.  
- **ArrayID** – Array job identifier (`-1` for non-array jobs).  
- **User** – Username of the job submitter.  
- **Account** – Account/group associated with the job.  

### Time Attributes
- **StartTime** – When the job started execution (datetime).  
- **SubmitTime** – When the job was submitted (datetime).  
- **Elapsed** – Total runtime duration (timedelta).  
- **TimeLimit** – Maximum allowed runtime (timedelta).  

### Resource Allocation
- **GPUs** – Number of GPUs allocated.  
- **GPUType** – Type of GPU allocated (e.g., `"v100"`, `"a100"`, or `NA` for CPU-only jobs).  
- **Nodes** – Number of nodes allocated.  
- **CPUs** – Number of CPU cores allocated.  
- **ReqMem** – Requested memory.  

### Job Status
- **Status** – Final job status (`"COMPLETED"`, `"FAILED"`, `"CANCELLED"`, etc.).  
- **ExitCode** – Job exit code.  
- **QOS** – Quality of Service level.  
- **Partition** – Cluster partition used.  

### Resource Usage
- **CPUTime** – Total CPU time used.  
- **CPUTimeRAW** – Raw CPU time measurement.  

### Constraints and Configuration
- **Constraints** – Hardware constraints specified.  
- **Interactive** – Whether the job was interactive (`"interactive"` or `"non-interactive"`).  

---

## Efficiency and Resource Metrics

### GPU and VRAM Metrics

- **GPU Count** (`gpu_count`)  
  Number of GPUs allocated to the job.

- **Job Hours** (`job_hours`)  
  $$
  \text{job\_hours} = \frac{\text{Elapsed (seconds)}}{3600} \times \text{gpu\_count}
  $$

- **VRAM Constraint** (`vram_constraint`)  
  VRAM requested via constraints, in GiB. Defaults are applied if not explicitly requested.

- **Partition Constraint** (`partition_constraint`)  
  VRAM derived from selecting a GPU partition, in GiB.

- **Requested VRAM** (`requested_vram`)  
  $$
  \text{requested\_vram} =
  \begin{cases}
    \text{partition\_constraint}, & \text{if available} \\
    \text{vram\_constraint}, & \text{otherwise}
  \end{cases}
  $$

- **Used VRAM** (`used_vram_gib`)  
  Sum of peak VRAM used on all allocated GPUs (GiB).

- **Approximate Allocated VRAM** (`allocated_vram`)  
  Estimated VRAM based on GPU model(s) and job node allocation.

- **Total VRAM-Hours** (`vram_hours`)  
  $$
  \text{vram\_hours} = \text{allocated\_vram} \times \text{job\_hours}
  $$

- **Allocated VRAM Efficiency** (`alloc_vram_efficiency`)  
  $$
  \text{alloc\_vram\_efficiency} = \frac{\text{used\_vram\_gib}}{\text{allocated\_vram}}
  $$

- **VRAM Constraint Efficiency** (`vram_constraint_efficiency`)  
  $$
  \text{vram\_constraint\_efficiency} =
  \frac{\text{used\_vram\_gib}}{\text{vram\_constraint}}
  $$

- **Allocated VRAM Efficiency Score** (`alloc_vram_efficiency_score`)  
  $$
  \text{alloc\_vram\_efficiency\_score} =
  \ln(\text{alloc\_vram\_efficiency}) \times \text{vram\_hours}
  $$
  Penalizes long jobs with low VRAM efficiency.

- **VRAM Constraint Efficiency Score** (`vram_constraint_efficiency_score`)  
  $$
  \text{vram\_constraint\_efficiency\_score} =
  \ln(\text{vram\_constraint\_efficiency}) \times \text{vram\_hours}
  $$

### CPU Memory Metrics
- **Used CPU Memory** (`used_cpu_mem_gib`) – Peak CPU RAM usage in GiB.  
- **Allocated CPU Memory** (`allocated_cpu_mem_gib`) – Requested CPU RAM in GiB.  
- **CPU Memory Efficiency** (`cpu_mem_efficiency`)  
  $$
  \text{cpu\_mem\_efficiency} = \frac{\text{used\_cpu\_mem\_gib}}{\text{allocated\_cpu\_mem\_gib}}
  $$

---

## User-Level Metrics

- **Job Count** (`job_count`) – Number of jobs submitted by the user.  
- **Total Job Hours** (`user_job_hours`) – Sum of job hours for all jobs of the user.  
- **Average Allocated VRAM Efficiency Score** (`avg_alloc_vram_efficiency_score`).  
- **Average VRAM Constraint Efficiency Score** (`avg_vram_constraint_efficiency_score`).  

- **Weighted Average Allocated VRAM Efficiency**  
  $$
  \text{expected\_value\_alloc\_vram\_efficiency} =
  \frac{\sum (\text{alloc\_vram\_efficiency} \times \text{vram\_hours})}
       {\sum \text{vram\_hours}}
  $$

- **Weighted Average VRAM Constraint Efficiency**  
  $$
  \text{expected\_value\_vram\_constraint\_efficiency} =
  \frac{\sum (\text{vram\_constraint\_efficiency} \times \text{vram\_hours})}
       {\sum \text{vram\_hours}}
  $$

- **Weighted Average GPU Count**  
  $$
  \text{expected\_value\_gpu\_count} =
  \frac{\sum (\text{gpu\_count} \times \text{vram\_hours})}
       {\sum \text{vram\_hours}}
  $$

- **Total VRAM-Hours** – Sum of allocated_vram × job_hours across all jobs of the user.

---

## Group-Level Metrics

For a group of users (e.g., PI group):

- **Job Count** – Total number of jobs across the group.  
- **PI Group Job Hours** (`pi_acc_job_hours`).  
- **PI Group VRAM Hours** (`pi_ac_vram_hours`).  
- **User Count**.  
- Group averages and weighted averages of efficiency metrics (similar formulas as above).

---

## Efficiency Categories
- **High**: > 70%  
- **Medium**: 30–70%  
- **Low**: 10–30%  
- **Very Low**: < 10%  
