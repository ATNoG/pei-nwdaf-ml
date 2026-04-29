"""Shared tag schema for filtering network data by slice/event."""

from pydantic import BaseModel, Field


class Tags(BaseModel):
    snssai_sst: str = Field(..., description="S-NSSAI SST (e.g. '1')")
    snssai_sd: str | None = Field(None, description="S-NSSAI SD (e.g. '000001')")
    dnn: str = Field(..., description="Data Network Name (e.g. 'internet')")
    event: str | None = Field(None, description="Event type (e.g. 'PERF_DATA')")

    def to_filter_dict(self) -> dict[str, str | None]:
        return {
            "snssai_sst": self.snssai_sst,
            "snssai_sd": self.snssai_sd,
            "dnn": self.dnn,
            "event": self.event,
        }
