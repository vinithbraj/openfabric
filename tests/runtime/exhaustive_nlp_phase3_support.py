from __future__ import annotations

import json
import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "exhaustive_nlp_phase3_cases.json"
WORKSPACE_NAME = "exhaustive_nlp_phase3"


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    category: str
    prompt: str
    expected: Any
    mode: str
    deterministic_expected: bool = True
    unsupported_reason: str | None = None


NOTES_CONTENT = {
    "meeting_notes.txt": "agenda\nbudget\nfollow-up\n",
    "shopping_list.txt": "bread\nmilk\ntea\n",
    "poem.txt": "roses\nviolets\nstarlight\n",
    "todo.txt": "email\ncall\nlunch\n",
    "welcome.txt": "hello\nworld\nkind\n",
    "travel.txt": "train\nhotel\nmuseum\n",
    "garden.txt": "seeds\nwater\nharvest\n",
    "library.txt": "fiction\nhistory\nscience\n",
    "kitchen.txt": "salt\npepper\ncinnamon\n",
    "journal.txt": "morning\nafternoon\nevening\n",
}

FILE_TREE = {
    "alpha": {"a.txt": "a\n", "b.txt": "b\n", "ignore.md": "ignore\n", "nested/c.txt": "c\n"},
    "beta": {"first.txt": "first\n", "second.txt": "second\n", "ignore.md": "ignore\n", "nested/third.txt": "third\n"},
    "gamma": {"blue.txt": "blue\n", "red.txt": "red\n", "ignore.md": "ignore\n", "nested/green.txt": "green\n"},
    "delta": {"north.txt": "north\n", "south.txt": "south\n", "ignore.md": "ignore\n", "nested/east.txt": "east\n"},
    "epsilon": {"apple.txt": "apple\n", "banana.txt": "banana\n", "ignore.md": "ignore\n", "nested/citrus.txt": "citrus\n"},
    "zeta": {"oak.txt": "oak\n", "pine.txt": "pine\n", "ignore.md": "ignore\n", "nested/cedar.txt": "cedar\n"},
    "eta": {"sun.txt": "sun\n", "moon.txt": "moon\n", "ignore.md": "ignore\n", "nested/star.txt": "star\n"},
    "theta": {"lake.txt": "lake\n", "river.txt": "river\n", "ignore.md": "ignore\n", "nested/ocean.txt": "ocean\n"},
    "iota": {"bread.txt": "bread\n", "soup.txt": "soup\n", "ignore.md": "ignore\n", "nested/tea.txt": "tea\n"},
    "kappa": {"alpha.txt": "alpha\n", "beta.txt": "beta\n", "ignore.md": "ignore\n", "nested/gamma.txt": "gamma\n"},
}

SEARCH_TREE = {
    "pantry/cake.txt": "cinnamon sugar\n",
    "pantry/tea.txt": "ginger and cinnamon\n",
    "pantry/salt.txt": "sea salt\n",
    "pantry/notes.md": "cinnamon markdown\n",
    "journal/april.txt": "garden plans\n",
    "journal/may.txt": "garden party\n",
    "journal/june.md": "garden sketch\n",
    "stories/library.txt": "quiet library afternoon\n",
    "stories/park.txt": "sunny park walk\n",
    "travel/lisbon.txt": "harbor light\n",
    "travel/oslo.txt": "snow path\n",
    "travel/rome.txt": "museum and harbor\n",
    "kitchen/bread.txt": "fresh bread\n",
    "kitchen/spice.txt": "tea and cinnamon\n",
    "kitchen/soup.txt": "ginger soup\n",
    "kitchen/notes.md": "cinnamon markdown\n",
    "weekend/garden.txt": "weekend garden\n",
    "weekend/market.txt": "weekend market\n",
    "weekend/desk.txt": "weekday notes\n",
    "puzzle/logic.txt": "puzzle pieces\n",
    "puzzle/riddle.txt": "puzzle answer\n",
    "puzzle/plain.txt": "sunrise walk\n",
    "orchard/basket.txt": "orchard basket\n",
    "orchard/trees.txt": "orchard apple\n",
    "orchard/shed.txt": "garden tools\n",
    "lantern/day.txt": "morning light\n",
    "lantern/night.txt": "lantern glow\n",
    "lantern/festival.txt": "lantern song\n",
    "cafe/bill.txt": "table three\n",
    "cafe/menu.txt": "tea and toast\n",
    "cafe/notes.txt": "cinnamon bun\n",
}


def load_cases(case_ids: set[str] | None = None) -> list[CaseSpec]:
    payload = json.loads(FIXTURE_PATH.read_text())
    cases = [
        CaseSpec(
            case_id=item["case_id"],
            category=item["category"],
            prompt=item["prompt"],
            expected=item["expected"],
            mode=item["mode"],
            deterministic_expected=bool(item.get("deterministic_expected", True)),
            unsupported_reason=item.get("unsupported_reason"),
        )
        for item in payload
    ]
    if case_ids is None:
        return cases
    return [case for case in cases if case.case_id in case_ids]


def rebuild_workspace(workspace: Path) -> dict[str, Any]:
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    for name, content in NOTES_CONTENT.items():
        _write(workspace / "notes" / name, content)

    for directory, files in FILE_TREE.items():
        for relative_path, content in files.items():
            _write(workspace / "files" / directory / relative_path, content)

    for relative_path, content in SEARCH_TREE.items():
        _write(workspace / "search" / relative_path, content)

    (workspace / "outputs").mkdir(parents=True, exist_ok=True)
    (workspace / "writes").mkdir(parents=True, exist_ok=True)

    sql_path = workspace / "book_club.db"
    database = sqlite3.connect(sql_path)
    try:
        database.executescript(
            """
            CREATE TABLE members (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                city TEXT NOT NULL
            );
            INSERT INTO members(id, name, city) VALUES
                (1, 'Alice', 'Portland'),
                (2, 'Bob', 'Seattle'),
                (3, 'Carla', 'Portland');
            """
        )
        database.commit()
    finally:
        database.close()

    return {
        "sql_databases": {"book_club_db": f"sqlite:///{sql_path}"},
        "sql_default_database": "book_club_db",
    }


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
