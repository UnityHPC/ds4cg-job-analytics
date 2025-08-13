import pytest

from src.config.enum_constants import (
    MetricsDataFrameNameBase,
    MetricsDataFrameNameEnum,
    ResourceHoardingDataFrameNameEnum,
)


def test_valid_subclass_creation_passes():
    class ValidMetricsEnum(MetricsDataFrameNameBase):
        JOBS = "jobs_with_efficiency_metrics"
        USERS = "users_with_efficiency_metrics"
        PI_GROUPS = "pi_accounts_with_efficiency_metrics"

    # Sanity checks
    assert ValidMetricsEnum.JOBS.value == "jobs_with_efficiency_metrics"
    assert ValidMetricsEnum.USERS.value == "users_with_efficiency_metrics"
    assert ValidMetricsEnum.PI_GROUPS.value == "pi_accounts_with_efficiency_metrics"


def test_missing_required_members_raises():
    with pytest.raises(TypeError, match="must define members"):

        class MissingMembersEnum(MetricsDataFrameNameBase):
            USERS = "users_with_efficiency_metrics"
            PI_GROUPS = "pi_accounts_with_efficiency_metrics"


def test_wrong_value_for_required_member_raises():
    with pytest.raises(TypeError, match=r"JOBS must equal 'jobs_with_efficiency_metrics'"):

        class WrongValueEnum(MetricsDataFrameNameBase):
            JOBS = "wrong_value"
            USERS = "users_with_efficiency_metrics"
            PI_GROUPS = "pi_accounts_with_efficiency_metrics"


def test_existing_enums_satisfy_contract():
    # The canonical enum should satisfy the metaclass contract exactly
    assert MetricsDataFrameNameEnum.JOBS.value == "jobs_with_efficiency_metrics"
    assert MetricsDataFrameNameEnum.USERS.value == "users_with_efficiency_metrics"
    assert MetricsDataFrameNameEnum.PI_GROUPS.value == "pi_accounts_with_efficiency_metrics"

    # The ResourceHoarding enum reuses canonical values and should also satisfy the contract
    assert ResourceHoardingDataFrameNameEnum.JOBS.value == "jobs_with_efficiency_metrics"
    assert ResourceHoardingDataFrameNameEnum.USERS.value == "users_with_efficiency_metrics"
    assert ResourceHoardingDataFrameNameEnum.PI_GROUPS.value == "pi_accounts_with_efficiency_metrics"
