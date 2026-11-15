from typing import Literal, Optional

from pydantic import BaseModel, Field


class Source(BaseModel):
    file: str = Field(description="File name from a tool result, e.g. kodiaq.md")
    lines: str = Field(description="Line range from that result, e.g. 16-21")


class FinalAnswer(BaseModel):
    answer: str = Field(min_length=1, description="The reply to show the user")
    sources: list[Source] = Field(default_factory=list, description="Passages the answer relies on. Empty when the reply has no factual claim")


class PlanStep(BaseModel):
    id: int
    goal: str
    status: Literal["todo", "done", "failed"] = "todo"


class Plan(BaseModel):
    """The full list of steps for this question. Call it to mark steps done or failed."""
    steps: list[PlanStep] = Field(min_length=1)

    def unfinished(self) -> list[PlanStep]:
        return [step for step in self.steps if step.status == "todo"]

    def render(self) -> str:
        marks = {"todo": "[ ]", "done": "[x]", "failed": "[!]"}
        return "\n".join(f"{marks[step.status]} {step.id}. {step.goal}" for step in self.steps)


class ProfileUpdate(BaseModel):
    field: Literal["budget", "body", "fuel", "seats", "use", "must_haves", "avoid", "other"]
    value: str
    quote: str = Field(description="Exact words from the user that support this")
    strength: Literal["soft", "hard"] = Field(
        default="soft",
        description="soft is a vague interest such as 'looking for'. hard is explicit: only, must, רק, חייב",
    )


class Triage(BaseModel):
    parts: list[str] = Field(min_length=1, description="Separate things the user wants. A simple question has exactly one part")
    kind: Literal["direct", "recommendation"] = Field(
        description="direct: a question about named cars. recommendation: they want a suggestion or a list",
    )
    profile_updates: list[ProfileUpdate] = Field(default_factory=list)
