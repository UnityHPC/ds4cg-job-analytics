# Zero GPU Usage Functions
### User data and outreach

The ```zero_gpu_usage_list.py``` script generates a list of users who have repeatedly failed
to utilize requested GPUs in their jobs, and have never sucessfully used it. It generates personalized 
email bodies with user-specific resource usage. This script will only run on Unity, for users part
of the ```pi_bpachev_umass_edu``` group. It is included as an example of the sort of tool that 
might be useful to the Unity team as a final deliverable of this project.

::: scripts.zero_gpu_usage_list