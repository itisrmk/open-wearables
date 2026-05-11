from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

from app.constants.sleep import SleepStageType
from app.database import DbSession
from app.repositories.user_connection_repository import UserConnectionRepository
from app.schemas.enums import HealthScoreCategory, ProviderName, SeriesType
from app.schemas.model_crud.activities import (
    EventRecordCreate,
    SleepStage,
    EventRecordDetailCreate,
    HealthScoreCreate,
    ScoreComponent,
    TimeSeriesSampleCreate,
)
from app.schemas.providers.polar import DailyActivityJSON, SleepJSON
from app.services.providers.api_client import make_authenticated_request
from app.services.providers.templates.base_247_data import Base247DataTemplate
from app.services.providers.templates.base_oauth import BaseOAuthTemplate


class Polar247Data(Base247DataTemplate):

    _HYPNOGRAM_STAGE_MAP: dict[int, SleepStageType] = {
        0: SleepStageType.AWAKE,
        1: SleepStageType.REM,
        2: SleepStageType.LIGHT,
        3: SleepStageType.LIGHT,
        4: SleepStageType.DEEP,
        5: SleepStageType.UNKNOWN,
    }

    def __init__(
        self,
        provider_name: str,
        api_base_url: str,
        oauth: BaseOAuthTemplate,
    ):
        super().__init__(provider_name, api_base_url, oauth)
        self.connection_repo = UserConnectionRepository()

    def _make_api_request(
        self,
        db: DbSession,
        user_id: UUID,
        endpoint: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        return make_authenticated_request(
            db=db,
            user_id=user_id,
            connection_repo=self.connection_repo,
            oauth=self.oauth,
            api_base_url=self.api_base_url,
            provider_name=self.provider_name,
            endpoint=endpoint,
            method="GET",
            params=params,
            headers=headers,
        )

    # -------------------------------------------------------------------------
    # Sleep - GET /v3/users/sleep, GET /v3/users/sleep/{date} and GET /v3/users/sleep/available
    # -------------------------------------------------------------------------

    def _get_available_sleep_dates(self, db: DbSession, user_id: UUID) -> set[date]:
        response = self._make_api_request(db, user_id, "/v3/users/sleep/available")
        nights = response.get("nights", []) if isinstance(response, dict) else []
        return {
            date.fromisoformat(night["date"])
            for night in nights
            if night.get("date")
        }

    def get_sleep_data(
        self,
        db: DbSession,
        user_id: UUID,
        start_time: datetime,
        end_time: datetime,
    ) -> list[dict[str, Any]]:
        date_range = {
            start_time.date() + timedelta(days=i)
            for i in range((end_time.date() - start_time.date()).days + 1)
        }
        available_dates = self._get_available_sleep_dates(db, user_id)
        sleep_data = []
        for d in date_range.intersection(available_dates):
            response = self._make_api_request(db, user_id, f"/v3/users/sleep/{d.isoformat()}")
            if response:
                sleep_data.append(response)
        return sleep_data

    def _parse_hypnogram(
        self,
        hypnogram: dict[str, int],
        sleep_start: datetime,
        sleep_end: datetime,
    ) -> list[SleepStage]:
        # Build (datetime, stage_val) in API order, handling midnight crossover
        entries: list[tuple[datetime, int]] = []
        current_date = sleep_start.date()
        prev_t: time | None = None
        for key, stage_val in hypnogram.items():
            h, m = map(int, key.split(":"))
            t = time(h, m)
            if prev_t is not None and t < prev_t:
                current_date += timedelta(days=1)
            entries.append((datetime.combine(current_date, t, tzinfo=sleep_start.tzinfo), stage_val))
            prev_t = t

        if not entries:
            return []

        # Group consecutive runs of the same stage into a single SleepStage
        stages: list[SleepStage] = []
        group_start, current_val = entries[0]

        for dt, stage_val in entries[1:]:
            if stage_val != current_val:
                stage_type = self._HYPNOGRAM_STAGE_MAP.get(current_val)
                if stage_type is not None:
                    stages.append(SleepStage(stage=stage_type, start_time=group_start, end_time=dt))
                group_start = dt
                current_val = stage_val

        stage_type = self._HYPNOGRAM_STAGE_MAP.get(current_val)
        if stage_type is not None:
            stages.append(SleepStage(stage=stage_type, start_time=group_start, end_time=sleep_end))
        return stages

    def normalize_sleep(  # type: ignore[override]
        self,
        raw_sleep: dict[str, Any],
        user_id: UUID,
    ) -> tuple[EventRecordCreate, EventRecordDetailCreate, HealthScoreCreate | None]:
        parsed = SleepJSON.model_validate(raw_sleep)
        sleep_id = uuid4()

        start_dt = datetime.fromisoformat(parsed.sleep_start_time) if parsed.sleep_start_time else None
        end_dt = datetime.fromisoformat(parsed.sleep_end_time) if parsed.sleep_end_time else None
        duration_seconds = int((end_dt - start_dt).total_seconds()) if start_dt and end_dt else None

        light_s = parsed.light_sleep or 0
        deep_s = parsed.deep_sleep or 0
        rem_s = parsed.rem_sleep or 0
        sleep_stages = (
            self._parse_hypnogram(parsed.hypnogram, start_dt, end_dt)
            if parsed.hypnogram and start_dt and end_dt
            else None
        )

        record = EventRecordCreate(
            id=sleep_id,
            category="sleep",
            type="sleep_session",
            source_name="Polar",
            device_model=parsed.device_id,
            duration_seconds=duration_seconds,
            start_datetime=start_dt,
            end_datetime=end_dt,
            provider=ProviderName.POLAR,
            user_id=user_id,
        )

        detail = EventRecordDetailCreate(
            record_id=sleep_id,
            sleep_total_duration_minutes=(light_s + deep_s + rem_s) // 60,
            sleep_time_in_bed_minutes=duration_seconds // 60 if duration_seconds else None,
            sleep_deep_minutes=deep_s // 60,
            sleep_light_minutes=light_s // 60,
            sleep_rem_minutes=rem_s // 60,
            sleep_awake_minutes=(parsed.total_interruption_duration or 0) // 60,
            sleep_stages=sleep_stages,
        )

        score: HealthScoreCreate | None = None
        if parsed.sleep_score is not None and start_dt is not None:
            raw_components: dict[str, float | int | None] = {
                "sleep_time": parsed.group_duration_score,
                "long_interruptions": parsed.long_interruption_duration,
                "continuity": parsed.continuity,
                "actual_sleep": parsed.group_solidity_score,
                "rem_sleep": parsed.rem_sleep,
                "deep_sleep": parsed.deep_sleep,
            }
            components: dict[str, ScoreComponent] = {
                k: ScoreComponent(value=v)
                for k, v in raw_components.items()
                if v is not None
            }
            score = HealthScoreCreate(
                id=uuid4(),
                user_id=user_id,
                provider=ProviderName.POLAR,
                category=HealthScoreCategory.SLEEP,
                value=parsed.sleep_score,
                recorded_at=start_dt,
                components=components or None,
                sleep_record_id=sleep_id,
            )

        return record, detail, score

    # -------------------------------------------------------------------------
    # Daily Activity - GET /v3/users/activities
    # -------------------------------------------------------------------------

    def get_daily_activity_statistics(
        self,
        db: DbSession,
        user_id: UUID,
        start_date: datetime,
        end_date: datetime,
    ) -> list[dict[str, Any]]:
        params = {
            "from": start_date.date().isoformat(),
            "to": end_date.date().isoformat(),
            "steps": "true",
            "activity_zones": "false",
            "inactivity_stamps": "false",
        }
        response = self._make_api_request(db, user_id, "/v3/users/activities", params=params)
        return response if isinstance(response, list) else []

    def normalize_daily_activity(  # type: ignore[override]
        self,
        raw_stats: dict[str, Any],
        user_id: UUID,
    ) -> list[TimeSeriesSampleCreate]:
        parsed = DailyActivityJSON.model_validate(raw_stats)
        if not parsed.start_time:
            return []

        recorded_at = datetime.fromisoformat(parsed.start_time)
        samples: list[TimeSeriesSampleCreate] = []

        if parsed.steps is not None:
            samples.append(TimeSeriesSampleCreate(
                id=uuid4(),
                user_id=user_id,
                provider=ProviderName.POLAR,
                source=ProviderName.POLAR,
                recorded_at=recorded_at,
                value=parsed.steps,
                series_type=SeriesType.steps,
            ))

        if parsed.active_calories is not None:
            samples.append(TimeSeriesSampleCreate(
                id=uuid4(),
                user_id=user_id,
                provider=ProviderName.POLAR,
                source=ProviderName.POLAR,
                recorded_at=recorded_at,
                value=parsed.active_calories,
                series_type=SeriesType.energy,
            ))

        if parsed.distance_from_steps is not None:
            samples.append(TimeSeriesSampleCreate(
                id=uuid4(),
                user_id=user_id,
                provider=ProviderName.POLAR,
                source=ProviderName.POLAR,
                recorded_at=recorded_at,
                value=Decimal(str(parsed.distance_from_steps)),
                series_type=SeriesType.distance_walking_running,
            ))

        return samples

    # -------------------------------------------------------------------------
    # Not implemented — Polar recovery and activity samples map to other modules
    # -------------------------------------------------------------------------

    def get_recovery_data(
        self,
        db: DbSession,
        user_id: UUID,
        start_time: datetime,
        end_time: datetime,
    ) -> list[dict[str, Any]]:
        return []

    def normalize_recovery(
        self,
        raw_recovery: dict[str, Any],
        user_id: UUID,
    ) -> dict[str, Any]:
        return {}

    def get_activity_samples(
        self,
        db: DbSession,
        user_id: UUID,
        start_time: datetime,
        end_time: datetime,
    ) -> list[dict[str, Any]]:
        return []

    def normalize_activity_samples(
        self,
        raw_samples: list[dict[str, Any]],
        user_id: UUID,
    ) -> dict[str, list[dict[str, Any]]]:
        return {}
