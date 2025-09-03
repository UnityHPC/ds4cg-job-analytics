from .efficiency_analysis import EfficiencyAnalysis
import pandas as pd
from src.config.remote_config import NodeInfoFetcher
from src.warnings import NodeNotFoundWarning
from src.config.enum_constants import (
    RequiredHoardingAnalysisColumnsEnum,
    NodeInfoKeyEnum,
    ResourceHoardingDataFrameNameEnum,
)
import warnings


class ResourceHoarding(EfficiencyAnalysis[ResourceHoardingDataFrameNameEnum]):
    """Analyze resource hoarding in jobs."""

    def __init__(self, jobs_df: pd.DataFrame) -> None:
        # Pass the subclass-specific enum to the base initializer
        super().__init__(jobs_df, metrics_df_name_enum=ResourceHoardingDataFrameNameEnum)

    def _get_resource_totals_for_job(
        self, jobs_node_list: list[str], node_info: list[dict]
    ) -> dict[NodeInfoKeyEnum, int | pd.api.typing.NAType]:
        """Get total resource values for a list of nodes.

        Args:
            jobs_node_list (list[str]): List of node names.
            node_info (list[dict]): Node information dictionary obtained from NodeInfoFetcher.

        Returns:
            dict[NodeInfoKeyEnum, int | pd.api.typing.NAType]:
                Dictionary containing total RAM, GPU count, and CPU core count.
        """
        if len(jobs_node_list) == 0:
            return {
                NodeInfoKeyEnum.RAM: pd.NA,
                NodeInfoKeyEnum.GPU_COUNT: pd.NA,
                NodeInfoKeyEnum.CORE_COUNT_PER_NODE: pd.NA,
            }

        ram_values = []
        gpu_values = []
        core_values = []
        for node in jobs_node_list:
            values = NodeInfoFetcher.get_node_info_values(
                node,
                node_info,
                members={
                    NodeInfoKeyEnum.GPU_COUNT,
                    NodeInfoKeyEnum.RAM,
                    NodeInfoKeyEnum.CORE_COUNT_PER_NODE,
                },
                offline=True,
            )
            ram_values.append(values[NodeInfoKeyEnum.RAM])
            gpu_values.append(values[NodeInfoKeyEnum.GPU_COUNT])
            core_values.append(values[NodeInfoKeyEnum.CORE_COUNT_PER_NODE])

        total_ram = pd.Series(ram_values, dtype=pd.Int64Dtype()).sum(skipna=False)
        total_gpu = pd.Series(gpu_values, dtype=pd.Int64Dtype()).sum(skipna=False)
        total_cores = pd.Series(core_values, dtype=pd.Int64Dtype()).sum(skipna=False)
        return {
            NodeInfoKeyEnum.RAM: total_ram,
            NodeInfoKeyEnum.GPU_COUNT: total_gpu,
            NodeInfoKeyEnum.CORE_COUNT_PER_NODE: total_cores,
        }

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
            total_available_cores_per_job = resource_totals_per_job.apply(
                lambda d: d[NodeInfoKeyEnum.CORE_COUNT_PER_NODE]
            ).astype(pd.Int32Dtype())

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
            NodeInfoKeyEnum.CORE_COUNT_PER_NODE: total_available_cores_per_job,
        }

    def calculate_node_resource_hoarding_for_jobs(self, filtered_jobs: pd.DataFrame) -> pd.DataFrame:
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
        resource_hoarding_jobs = self.calculate_job_efficiency_metrics(filtered_jobs)

        # check if cpu_mem_efficiency and used_cpu_mem_gib and allocated_cpu_mem_gib are present
        missing_columns = [
            key.value
            for key in RequiredHoardingAnalysisColumnsEnum.__members__.values()
            if key.value not in resource_hoarding_jobs.columns
        ]
        if len(missing_columns) > 0:
            raise ValueError(
                f"Missing required CPU memory efficiency metrics: "
                f"{', '.join(missing_columns)}. "
                "CPU-related metrics are required for analysis."
            )

        total_node_resources_per_job = self._calculate_total_resources_of_nodes_per_job(resource_hoarding_jobs)

        # Add memory hoarding metrics
        resource_hoarding_jobs.loc[:, "total_ram_of_nodes_gib"] = total_node_resources_per_job[NodeInfoKeyEnum.RAM]
        resource_hoarding_jobs.loc[:, "total_gpu_count_of_nodes"] = total_node_resources_per_job[
            NodeInfoKeyEnum.GPU_COUNT
        ]
        resource_hoarding_jobs.loc[:, "gpu_count_fraction"] = (
            resource_hoarding_jobs.loc[:, "gpu_count"] / resource_hoarding_jobs.loc[:, "total_gpu_count_of_nodes"]
        )

        resource_hoarding_jobs.loc[:, "allocated_ram_fraction"] = (
            resource_hoarding_jobs.loc[:, "allocated_cpu_mem_gib"]
            / resource_hoarding_jobs.loc[:, "total_ram_of_nodes_gib"]
        )

        resource_hoarding_jobs.loc[:, "ram_hoarding_fraction_diff"] = (
            resource_hoarding_jobs.loc[:, "allocated_ram_fraction"]
            - resource_hoarding_jobs.loc[:, "gpu_count_fraction"]
        )

        # Add CPU core hoarding metrics
        resource_hoarding_jobs.loc[:, "total_cores_of_nodes"] = total_node_resources_per_job[
            NodeInfoKeyEnum.CORE_COUNT_PER_NODE
        ]
        resource_hoarding_jobs.loc[:, "allocated_cores_fraction"] = (
            resource_hoarding_jobs.loc[:, "cpu_core_count"] / resource_hoarding_jobs.loc[:, "total_cores_of_nodes"]
        )

        resource_hoarding_jobs.loc[:, "core_hoarding_fraction_diff"] = (
            resource_hoarding_jobs.loc[:, "allocated_cores_fraction"]
            - resource_hoarding_jobs.loc[:, "gpu_count_fraction"]
        )

        self.jobs_with_resource_hoarding_metrics = resource_hoarding_jobs
        return self.jobs_with_resource_hoarding_metrics

    def calculate_node_resource_hoarding_for_users(self, filtered_jobs: pd.DataFrame) -> pd.DataFrame:
        """Calculate resource hoarding for users based on jobs with resource hoarding metrics.

        Args:
            filtered_jobs (pd.DataFrame): DataFrame containing jobs to analyze.

        Returns:
            pd.DataFrame: DataFrame with user-level resource hoarding metrics.
        """
        if self.jobs_with_resource_hoarding_metrics is None:
            self.calculate_node_resource_hoarding_for_jobs(filtered_jobs)
            print(
                "Jobs DataFrame with resource hoarding metrics was not available. "
                "Calculated it using the filtered_jobs DataFrame."
            )

        if self.users_with_efficiency_metrics is None:
            self.calculate_user_efficiency_metrics()
            print(
                "Users DataFrame with efficiency metrics was not available. "
                "Calculated it using the filtered_jobs DataFrame."
            )

        user_vram_hours_per_job = self.jobs_with_resource_hoarding_metrics.groupby("User", observed=True)[
            "vram_hours"
        ].transform("sum")

        users_w_resource_hoarding_metrics = self.users_with_efficiency_metrics.copy()

        self.jobs_with_resource_hoarding_metrics.loc[:, "weighted_ram_hoarding_fraction_diff"] = (
            self.jobs_with_resource_hoarding_metrics["ram_hoarding_fraction_diff"]
            * self.jobs_with_resource_hoarding_metrics["vram_hours"]
            / user_vram_hours_per_job
        )

        users_w_resource_hoarding_metrics.loc[:, "expected_value_ram_hoarding_fraction_diff"] = (
            self.jobs_with_resource_hoarding_metrics.groupby("User", observed=True)[
                "weighted_ram_hoarding_fraction_diff"
            ]
            .apply(lambda series: series.sum() if not series.isna().all() else pd.NA)
            .to_numpy()
        )

        self.jobs_with_resource_hoarding_metrics.loc[:, "weighted_core_hoarding_fraction_diff"] = (
            self.jobs_with_resource_hoarding_metrics["core_hoarding_fraction_diff"]
            * self.jobs_with_resource_hoarding_metrics["vram_hours"]
            / user_vram_hours_per_job
        )

        users_w_resource_hoarding_metrics.loc[:, "expected_value_core_hoarding_fraction_diff"] = (
            self.jobs_with_resource_hoarding_metrics.groupby("User", observed=True)[
                "weighted_core_hoarding_fraction_diff"
            ]
            .apply(lambda series: series.sum() if not series.isna().all() else pd.NA)
            .to_numpy()
        )

        self.jobs_with_resource_hoarding_metrics = self.jobs_with_resource_hoarding_metrics.drop(
            columns=["weighted_ram_hoarding_fraction_diff", "weighted_core_hoarding_fraction_diff"]
        )

        self.users_with_resource_hoarding_metrics = users_w_resource_hoarding_metrics
        return self.users_with_resource_hoarding_metrics
