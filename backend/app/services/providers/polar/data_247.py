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
    EventRecordDetailCreate,
    HealthScoreCreate,
    ScoreComponent,
    SleepStage,
    TimeSeriesSampleCreate,
)
from app.schemas.providers.polar import CardioLoadJSON, ContinuousHeartRateJSON, DailyActivityJSON, NightlyRechargeJSON, SleepJSON
from app.services.providers.api_client import make_authenticated_request
from app.services.providers.templates.base_247_data import Base247DataTemplate
from app.services.providers.templates.base_oauth import BaseOAuthTemplate


class Polar247Data(Base247DataTemplate):
    _NIGHTLY_RECHARGE_STATUS_LABELS: dict[int, str] = {
        1: "very poor",
        2: "poor",
        3: "compromised",
        4: "ok",
        5: "good",
        6: "very good",
    }

    _ANS_CHARGE_STATUS_LABELS: dict[int, str] = {
        1: "much below usual",
        2: "below usual",
        3: "usual",
        4: "above usual",
        5: "much above usual",
    }

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
        return {date.fromisoformat(night["date"]) for night in nights if night.get("date")}

    def get_sleep_data(
        self,
        db: DbSession,
        user_id: UUID,
        start_time: datetime,
        end_time: datetime,
    ) -> list[dict[str, Any]]:
        date_range = {
            start_time.date() + timedelta(days=i) for i in range((end_time.date() - start_time.date()).days + 1)
        }
        available_dates = self._get_available_sleep_dates(db, user_id)
        sleep_data = []
        for d in date_range.intersection(available_dates):
            response = self._make_api_request(db, user_id, f"/v3/users/sleep/{d.isoformat()}")
            if response:
                sleep_data.append(response)
        return sleep_data

    def _parse_time_key(self, key: str) -> time:
        parts = key.split(":")
        h, m = int(parts[0]), int(parts[1])
        s = int(parts[2]) if len(parts) > 2 else 0
        return time(h, m, s)

    def _hhmm_to_datetimes(
        self,
        items: dict[str, Any],
        anchor: datetime,
    ) -> list[tuple[datetime, Any]]:
        """Convert dict[HH:MM or HH:MM:SS, value] to [(datetime, value)], handling midnight crossover."""
        result: list[tuple[datetime, Any]] = []
        current_date = anchor.date()
        prev_t: time | None = None
        for key, val in items.items():
            t = self._parse_time_key(key)
            if prev_t is not None and t < prev_t:
                current_date += timedelta(days=1)
            result.append((datetime.combine(current_date, t, tzinfo=anchor.tzinfo), val))
            prev_t = t
        return result

    def _parse_hypnogram(
        self,
        hypnogram: dict[str, int],
        sleep_start: datetime,
        sleep_end: datetime,
    ) -> list[SleepStage]:
        entries = self._hhmm_to_datetimes(hypnogram, sleep_start)
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

    def _parse_sleep_hr_samples(
        self,
        hr_samples: dict[str, int],
        sleep_start: datetime,
        user_id: UUID,
    ) -> list[TimeSeriesSampleCreate]:
        return [
            TimeSeriesSampleCreate(
                id=uuid4(),
                user_id=user_id,
                provider=ProviderName.POLAR,
                source=ProviderName.POLAR,
                recorded_at=dt,
                value=bpm,
                series_type=SeriesType.heart_rate,
            )
            for dt, bpm in self._hhmm_to_datetimes(hr_samples, sleep_start)
        ]

    def normalize_sleep(  # type: ignore[override]
        self,
        raw_sleep: dict[str, Any],
        user_id: UUID,
    ) -> tuple[
            EventRecordCreate,
            EventRecordDetailCreate,
            HealthScoreCreate | None,
            list[TimeSeriesSampleCreate]
        ]:
        parsed = SleepJSON.model_validate(raw_sleep)
        sleep_id = uuid4()

        if not parsed.sleep_start_time or not parsed.sleep_end_time:
            raise ValueError(f"Polar sleep record missing start/end time: {parsed.date}")
        start_dt = datetime.fromisoformat(parsed.sleep_start_time)
        end_dt = datetime.fromisoformat(parsed.sleep_end_time)
        duration_seconds = int((end_dt - start_dt).total_seconds())

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
        if parsed.sleep_score is not None:
            raw_components: dict[str, float | int | None] = {
                "sleep_time": parsed.group_duration_score,
                "long_interruptions": parsed.long_interruption_duration,
                "continuity": parsed.continuity,
                "actual_sleep": parsed.group_solidity_score,
                "rem_sleep": parsed.rem_sleep,
                "deep_sleep": parsed.deep_sleep,
            }
            components: dict[str, ScoreComponent] = {
                k: ScoreComponent(value=v) for k, v in raw_components.items() if v is not None
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

        hr_samples = (
            self._parse_sleep_hr_samples(parsed.heart_rate_samples, start_dt, user_id)
            if parsed.heart_rate_samples
            else []
        )

        return record, detail, score, hr_samples

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
            samples.append(
                TimeSeriesSampleCreate(
                    id=uuid4(),
                    user_id=user_id,
                    provider=ProviderName.POLAR,
                    source=ProviderName.POLAR,
                    recorded_at=recorded_at,
                    value=parsed.steps,
                    series_type=SeriesType.steps,
                )
            )

        if parsed.active_calories is not None:
            samples.append(
                TimeSeriesSampleCreate(
                    id=uuid4(),
                    user_id=user_id,
                    provider=ProviderName.POLAR,
                    source=ProviderName.POLAR,
                    recorded_at=recorded_at,
                    value=parsed.active_calories,
                    series_type=SeriesType.energy,
                )
            )

        if parsed.distance_from_steps is not None:
            samples.append(
                TimeSeriesSampleCreate(
                    id=uuid4(),
                    user_id=user_id,
                    provider=ProviderName.POLAR,
                    source=ProviderName.POLAR,
                    recorded_at=recorded_at,
                    value=Decimal(str(parsed.distance_from_steps)),
                    series_type=SeriesType.distance_walking_running,
                )
            )

        return samples

    # -------------------------------------------------------------------------
    # Continuous Heart Rate - GET /v3/users/continuous-heart-rate/{date}
    # -------------------------------------------------------------------------

    def get_continuous_hr_data(
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
        results = []
        for d in date_range:
            response = self._make_api_request(
                db, user_id, f"/v3/users/continuous-heart-rate/{d.isoformat()}"
            )
            if response:
                results.append(response)
        return results

    def normalize_continuous_hr(
        self,
        raw: dict[str, Any],
        user_id: UUID,
    ) -> list[TimeSeriesSampleCreate]:
        parsed = ContinuousHeartRateJSON.model_validate(raw)
        if not parsed.date or not parsed.heart_rate_samples:
            return []

        anchor = datetime.fromisoformat(parsed.date)
        samples_dict = {s.sample_time: s.heart_rate for s in parsed.heart_rate_samples if s.sample_time}
        return [
            TimeSeriesSampleCreate(
                id=uuid4(),
                user_id=user_id,
                provider=ProviderName.POLAR,
                source=ProviderName.POLAR,
                recorded_at=dt,
                value=bpm,
                series_type=SeriesType.heart_rate,
            )
            for dt, bpm in self._hhmm_to_datetimes(samples_dict, anchor)
        ]

    # -------------------------------------------------------------------------
    # Cardio Load - GET /v3/users/cardio-load/{date}
    # -------------------------------------------------------------------------

    def get_cardio_load_data(
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
        results = []
        for d in date_range:
            response = self._make_api_request(db, user_id, f"/v3/users/cardio-load/{d.isoformat()}")
            if response:
                results.append(response)
        return results

    def normalize_cardio_load(
        self,
        raw: dict[str, Any],
        user_id: UUID,
    ) -> HealthScoreCreate | None:
        parsed = CardioLoadJSON.model_validate(raw)
        if parsed.cardio_load is None or not parsed.date:
            return None

        raw_components: dict[str, float | int | None] = {
            "strain": parsed.strain,
            "tolerance": parsed.tolerance,
            "cardio_load_ratio": parsed.cardio_load_ratio,
        }
        if parsed.cardio_load_level:
            lvl = parsed.cardio_load_level
            raw_components.update({
                "level_very_low": lvl.very_low,
                "level_low": lvl.low,
                "level_medium": lvl.medium,
                "level_high": lvl.high,
                "level_very_high": lvl.very_high,
            })
        components = {k: ScoreComponent(value=v) for k, v in raw_components.items() if v is not None}

        return HealthScoreCreate(
            id=uuid4(),
            user_id=user_id,
            provider=ProviderName.POLAR,
            category=HealthScoreCategory.STRAIN,
            value=parsed.cardio_load,
            recorded_at=datetime.fromisoformat(parsed.date),
            components=components or None,
        )

    # -------------------------------------------------------------------------
    # Nightly Recharge - GET /v3/users/nightly-recharge/{date}
    # -------------------------------------------------------------------------

    def get_nightly_recharge_data(
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
        results = []
        for d in date_range:
            response = self._make_api_request(db, user_id, f"/v3/users/nightly-recharge/{d.isoformat()}")
            if response:
                results.append(response)
        return results

    def normalize_nightly_recharge(
        self,
        raw: dict[str, Any],
        user_id: UUID,
    ) -> HealthScoreCreate | None:
        parsed = NightlyRechargeJSON.model_validate(raw)
        if parsed.nightly_recharge_status is None or not parsed.date:
            return None

        components: dict[str, ScoreComponent] = {}
        for key, val in {
            "heart_rate_avg": parsed.heart_rate_avg,
            "beat_to_beat_avg": parsed.beat_to_beat_avg,
            "heart_rate_variability_avg": parsed.heart_rate_variability_avg,
            "breathing_rate_avg": parsed.breathing_rate_avg,
            "ans_charge": parsed.ans_charge,
        }.items():
            if val is not None:
                components[key] = ScoreComponent(value=val)

        if parsed.ans_charge_status is not None:
            components["ans_charge_status"] = ScoreComponent(
                value=parsed.ans_charge_status,
                qualifier=self._ANS_CHARGE_STATUS_LABELS.get(parsed.ans_charge_status),
            )

        return HealthScoreCreate(
            id=uuid4(),
            user_id=user_id,
            provider=ProviderName.POLAR,
            category=HealthScoreCategory.RECOVERY,
            value=parsed.nightly_recharge_status,
            qualifier=self._NIGHTLY_RECHARGE_STATUS_LABELS.get(parsed.nightly_recharge_status),
            recorded_at=datetime.fromisoformat(parsed.date),
            components=components or None,
        )

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
