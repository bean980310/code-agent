"""Plan creation and tracking."""

from __future__ import annotations

from ..types import Plan, PlanStep


class Planner:
    """Manages multi-step plans for complex tasks."""

    def __init__(self) -> None:
        self.current_plan: Plan | None = None

    def create_plan(self, goal: str, steps: list[str]) -> Plan:
        """Create a new plan with the given goal and steps."""
        self.current_plan = Plan(
            goal=goal,
            steps=[PlanStep(id=i + 1, description=s) for i, s in enumerate(steps)],
        )
        return self.current_plan

    def update_step(self, step_id: int, status: str, result: str | None = None) -> None:
        """Update a step's status and optional result."""
        if self.current_plan is None:
            return
        for step in self.current_plan.steps:
            if step.id == step_id:
                step.status = status
                if result is not None:
                    step.result = result
                break

    def get_next_step(self) -> PlanStep | None:
        """Get the next pending step."""
        if self.current_plan is None:
            return None
        for step in self.current_plan.steps:
            if step.status == "pending":
                return step
        return None

    def is_complete(self) -> bool:
        """Check if all steps are done or failed."""
        if self.current_plan is None:
            return True
        return all(s.status in ("done", "failed") for s in self.current_plan.steps)

    def clear(self) -> None:
        """Discard the current plan."""
        self.current_plan = None

    def to_text(self) -> str:
        """Render the plan as readable text."""
        if self.current_plan is None:
            return "(no plan)"
        markers = {
            "pending": "[ ]",
            "in_progress": "[>]",
            "done": "[x]",
            "failed": "[!]",
        }
        lines = [f"Goal: {self.current_plan.goal}"]
        for step in self.current_plan.steps:
            m = markers.get(step.status, "[ ]")
            lines.append(f"  {m} {step.id}. {step.description}")
            if step.result:
                lines.append(f"       -> {step.result}")
        return "\n".join(lines)
