from .efficiency_analysis import EfficiencyAnalysis
import pandas as pd
from src.config.remote_config import NodeInfoFetcher
from src.warnings import NodeNotFoundWarning
from src.config.enum_constants import RequiredHoardingAnalysisColumnsEnum, NodeInfoKeyEnum
import warnings


class ResourceHoarding(EfficiencyAnalysis):
    """Analyze resource hoarding in jobs."""

    def __init__(self, jobs_df: pd.DataFrame) -> None:
        super().__init__(jobs_df)

    # TODO(Arda): Implement CPU core hoarding analysis
    # def calculate_cpu_core_hoarding(self) -> pd.DataFrame:
    #     return

    def _get_resource_totals_for_job(
        self, jobs_node_list: list[str], node_info: list[dict]
    ) -> dict[NodeInfoKeyEnum, int | pd.api.typing.NAType]:
        """Get total resource values for a list of nodes.

        Args:
            jobs_node_list (list[str]): List of node names.
            node_info (list[dict]): Node information dictionary obtained from NodeInfoFetcher.

        Returns:
            dict[NodeInfoKeyEnum, int | pd.api.typing.NAType]: Dictionary containing total RAM and GPU count.
        """
        if len(jobs_node_list) == 0:
            return {NodeInfoKeyEnum.RAM: pd.NA, NodeInfoKeyEnum.GPU_COUNT: pd.NA}

        ram_values = []
        gpu_values = []
        for node in jobs_node_list:
            values = NodeInfoFetcher.get_node_info_values(
                node, node_info, members={NodeInfoKeyEnum.GPU_COUNT, NodeInfoKeyEnum.RAM}, offline=True
            )
            ram_values.append(values[NodeInfoKeyEnum.RAM])
            gpu_values.append(values[NodeInfoKeyEnum.GPU_COUNT])

        total_ram = pd.Series(ram_values, dtype=pd.Int64Dtype()).sum(skipna=False)
        total_gpu = pd.Series(gpu_values, dtype=pd.Int64Dtype()).sum(skipna=False)
        return {NodeInfoKeyEnum.RAM: total_ram, NodeInfoKeyEnum.GPU_COUNT: total_gpu}

    def _calculate_total_resources_of_nodes_per_job(
        self, memory_hoarding_jobs: pd.DataFrame
    ) -> dict[NodeInfoKeyEnum, pd.Series]:
        """Calculate total available resources for nodes assigned to each job.

        Args:
            memory_hoarding_jobs (pd.DataFrame): DataFrame containing memory hoarding jobs.

        Returns:
            pd.Series: Series containing total available resources for each job.
        """
        node_info = NodeInfoFetcher().get_info()

        missing_nodes: set[str] = set()
        with warnings.catch_warnings(record=True) as node_warnings:
            warnings.simplefilter("default")
            resource_totals_per_job = memory_hoarding_jobs["NodeList"].apply(
                lambda node_list: self._get_resource_totals_for_job(node_list, node_info)
            )
            total_available_ram_per_job = resource_totals_per_job.apply(lambda d: d[NodeInfoKeyEnum.RAM]).astype(
                pd.Int32Dtype()
            )
            total_available_gpu_per_job = resource_totals_per_job.apply(lambda d: d[NodeInfoKeyEnum.GPU_COUNT]).astype(
                pd.Int32Dtype()
            )

            # Extract node names from warning messages
            for warn in node_warnings:
                if (
                    "not found in node configuration file" in str(warn.message)
                    and type(warn.category) is NodeNotFoundWarning
                    and hasattr(warn.message, "node_name")
                ):
                    node_name = warn.message.node_name
                    missing_nodes.add(node_name)
        if len(missing_nodes) > 0:
            warnings.warn(
                f"Missing node information for nodes: {', '.join(missing_nodes)}. "
                "This may affect the accuracy of memory hoarding analysis.",
                UserWarning,
                stacklevel=2,
            )

        return {
            NodeInfoKeyEnum.RAM: total_available_ram_per_job,
            NodeInfoKeyEnum.GPU_COUNT: total_available_gpu_per_job,
        }

    def calculate_memory_hoarding(self, filtered_jobs: pd.DataFrame) -> pd.DataFrame:
        """Detect memory hoarding in each job

        Checks if the ratio of requested memory to the available memory in each node is larger than
        the ratio of GPUs allocated to the number of GPUs available in each node. Raises warnings if
        any nodes are not found in the configuration.

        Args:
            filtered_jobs (pd.DataFrame): DataFrame containing jobs to analyze.

        Raises:
            ValueError: If required memory metrics are missing.

        Returns:
            pd.DataFrame: DataFrame with hoarding information for each job.
        """
        memory_hoarding_jobs = self.calculate_job_efficiency_metrics(filtered_jobs)
        # check if cpu_mem_efficiency and used_cpu_mem_gib and allocated_cpu_mem_gib are present
        missing_columns = [
            key.value
            for key in RequiredHoardingAnalysisColumnsEnum.__members__.values()
            if key.value not in memory_hoarding_jobs.columns
        ]

        if len(missing_columns) > 0:
            raise ValueError(
                f"Missing required CPU memory efficiency metrics: "
                f"{', '.join(missing_columns)}. "
                "CPU-related metrics are required for analysis."
            )
        total_node_resources_per_job = self._calculate_total_resources_of_nodes_per_job(memory_hoarding_jobs)
        memory_hoarding_jobs.loc[:, "total_ram_of_nodes_gib"] = total_node_resources_per_job[NodeInfoKeyEnum.RAM]
        memory_hoarding_jobs.loc[:, "total_gpu_count_of_nodes"] = total_node_resources_per_job[
            NodeInfoKeyEnum.GPU_COUNT
        ]
        memory_hoarding_jobs.loc[:, "gpu_count_fraction"] = (
            memory_hoarding_jobs.loc[:, "gpu_count"]
            / memory_hoarding_jobs.loc[:, "total_gpu_count_of_nodes"]
        )

        memory_hoarding_jobs.loc[:, "allocated_ram_fraction"] = (
            memory_hoarding_jobs.loc[:, "allocated_cpu_mem_gib"]
            / memory_hoarding_jobs.loc[:, "total_ram_of_nodes_gib"]
        )

        memory_hoarding_jobs.loc[:, "ram_hoarding_fraction_diff"] = (
            memory_hoarding_jobs.loc[:, "allocated_ram_fraction"]
            - memory_hoarding_jobs.loc[:, "gpu_count_fraction"]
        )

        self.jobs_with_efficiency_metrics = memory_hoarding_jobs
        return self.jobs_with_efficiency_metrics
