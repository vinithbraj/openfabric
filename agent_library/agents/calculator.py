import re

from fastapi import FastAPI

from agent_library.common import EventRequest, EventResponse

app = FastAPI()

AGENT_METADATA = {
    "description": "Performs arithmetic tasks from task plans and emits task results.",
    "routing_notes": [
        "Use for arithmetic word problems and math expressions.",
        "Do not use for filesystem or shell operations.",
    ],
    "methods": [
        {
            "name": "compute_basic_math",
            "event": "task.plan",
            "when": "Computes add/subtract/multiply/divide from natural-language task plans.",
            "intent_tags": ["math", "arithmetic"],
            "examples": ["add 12 and 30", "what is 144 / 12"],
            "anti_patterns": ["find files named vinith", "read README.md"],
        }
    ],
}


def _normalize_number(value: float):
    return int(value) if value.is_integer() else value


def _detect_operation(task: str):
    task_lc = task.lower()
    if (
        "divide" in task_lc
        or "divided by" in task_lc
        or " over " in task_lc
        or "/" in task
    ):
        return "divide"
    if (
        "multiply" in task_lc
        or "multipy" in task_lc
        or "mutiply" in task_lc
        or "times" in task_lc
        or re.search(r"\d\s*\*\s*-?\d", task)
    ):
        return "multiply"
    if (
        "subtract" in task_lc
        or "minus" in task_lc
        or re.search(r"\d\s*-\s*-?\d", task)
    ):
        return "subtract"
    if (
        "add" in task_lc
        or "sum" in task_lc
        or "plus" in task_lc
        or re.search(r"\d\s*\+\s*-?\d", task)
    ):
        return "add"
    return None


def _compute_basic_math(task: str):
    numbers = [float(item) for item in re.findall(r"-?\d+(?:\.\d+)?", task)]
    if len(numbers) < 2:
        return None

    operation = _detect_operation(task)
    if operation is None:
        return None

    if operation == "add":
        value = sum(numbers)
        return ("sum", _normalize_number(value))

    if operation == "subtract":
        value = numbers[0]
        for num in numbers[1:]:
            value -= num
        return ("difference", _normalize_number(value))

    if operation == "multiply":
        value = numbers[0]
        for num in numbers[1:]:
            value *= num
        return ("product", _normalize_number(value))

    if operation == "divide":
        value = numbers[0]
        for num in numbers[1:]:
            if num == 0:
                return ("error", "Cannot divide by zero")
            value /= num
        return ("quotient", _normalize_number(value))

    return None


@app.post("/handle", response_model=EventResponse)
def handle_event(req: EventRequest):
    if req.event == "task.plan":
        task = req.payload["task"]
        result = _compute_basic_math(task)

        if result is None:
            return {"emits": []}

        kind, value = result
        if kind == "error":
            detail = str(value)
        else:
            detail = f"Computed {kind} from task: {value}"

        return {"emits": [{"event": "task.result", "payload": {"detail": detail}}]}

    return {"emits": []}
