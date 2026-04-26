from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class EvalFixturePayload(BaseModel):
    workspace_root: str
    run_store_path: str
    sql_databases: dict[str, str] = Field(default_factory=dict)
    sql_default_database: str | None = None
    variables: dict[str, str] = Field(default_factory=dict)

    def settings_payload(self) -> dict[str, Any]:
        return {
            "workspace_root": self.workspace_root,
            "run_store_path": self.run_store_path,
            "sql_databases": dict(self.sql_databases),
            "sql_default_database": self.sql_default_database,
        }


NOTES_CONTENT = {
    "meeting_notes.txt": "agenda\nbudget\nfollow-up\n",
    "shopping.txt": "bread\nmilk\ntea\n",
    "poem.txt": "roses\nviolets\nstarlight\n",
    "todo.txt": "email\ncall\nlunch\n",
    "short.txt": "one\ntwo\n",
    "transform.txt": "quiet library\nLOUD VOICE\nmixed case words\nalpha beta\n",
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

FETCH_FIXTURES = {
    "example.html": '<html><head><title>Example Fixture</title><meta name="fixture" content="example"></head><body>example body</body></html>',
    "story.html": '<html><head><title>Story Fixture</title><meta name="fixture" content="story"></head><body>story body</body></html>',
    "museum.html": '<html><head><title>Museum Fixture</title><meta name="fixture" content="museum"></head><body>museum body</body></html>',
}


def rebuild_eval_workspace(workspace: Path) -> EvalFixturePayload:
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    notes_dir = workspace / "notes"
    for name, content in NOTES_CONTENT.items():
        _write(notes_dir / name, content)

    files_dir = workspace / "files"
    for directory, files in FILE_TREE.items():
        for relative_path, content in files.items():
            _write(files_dir / directory / relative_path, content)

    search_dir = workspace / "search"
    for relative_path, content in SEARCH_TREE.items():
        _write(search_dir / relative_path, content)

    outputs_dir = workspace / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    writes_dir = workspace / "writes"
    writes_dir.mkdir(parents=True, exist_ok=True)

    fetch_dir = workspace / "fetch"
    for name, content in FETCH_FIXTURES.items():
        _write(fetch_dir / name, content)

    sql_path = workspace / "book_club.db"
    db = sqlite3.connect(sql_path)
    try:
        db.executescript(
            """
            CREATE TABLE members (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                city TEXT NOT NULL
            );
            CREATE TABLE books (
                id INTEGER PRIMARY KEY,
                title TEXT NOT NULL
            );
            INSERT INTO members(id, name, city) VALUES
                (1, 'Alice', 'Portland'),
                (2, 'Bob', 'Seattle'),
                (3, 'Carla', 'Portland');
            INSERT INTO books(id, title) VALUES
                (1, 'North Window'),
                (2, 'Evening Train');
            """
        )
        db.commit()
    finally:
        db.close()

    variables: dict[str, str] = {
        "workspace_root": str(workspace),
        "outputs_dir": str(outputs_dir),
        "writes_dir": str(writes_dir),
        "search_root": str(search_dir),
    }
    variables.update(_path_variables("notes", notes_dir, NOTES_CONTENT))
    variables.update(_directory_variables("files", files_dir, FILE_TREE))
    variables.update(
        {
            "search_pantry": str(search_dir / "pantry"),
            "search_journal": str(search_dir / "journal"),
            "search_stories": str(search_dir / "stories"),
            "search_travel": str(search_dir / "travel"),
            "search_kitchen": str(search_dir / "kitchen"),
            "search_weekend": str(search_dir / "weekend"),
            "search_puzzle": str(search_dir / "puzzle"),
            "search_orchard": str(search_dir / "orchard"),
            "search_lantern": str(search_dir / "lantern"),
            "search_cafe": str(search_dir / "cafe"),
        }
    )
    variables.update(
        {
            "fetch_example": (fetch_dir / "example.html").resolve().as_uri(),
            "fetch_story": (fetch_dir / "story.html").resolve().as_uri(),
            "fetch_museum": (fetch_dir / "museum.html").resolve().as_uri(),
        }
    )

    return EvalFixturePayload(
        workspace_root=str(workspace),
        run_store_path=str(workspace / "runtime.db"),
        sql_databases={"book_club_db": f"sqlite:///{sql_path}"},
        sql_default_database="book_club_db",
        variables=variables,
    )


def render_template(value: Any, fixtures: EvalFixturePayload) -> Any:
    return _render_value(value, fixtures.variables)


def render_case_prompt(prompt: str, fixtures: EvalFixturePayload) -> str:
    return str(render_template(prompt, fixtures))


def render_case_expected(expected: Any, fixtures: EvalFixturePayload) -> Any:
    return render_template(expected, fixtures)


def _path_variables(prefix: str, root: Path, files: dict[str, str]) -> dict[str, str]:
    variables: dict[str, str] = {}
    for filename in files:
        stem = Path(filename).stem
        variables[f"{prefix}_{stem}"] = str(root / filename)
    return variables


def _directory_variables(prefix: str, root: Path, directories: dict[str, Any]) -> dict[str, str]:
    return {f"{prefix}_{name}": str(root / name) for name in directories}


def _render_value(value: Any, variables: dict[str, str]) -> Any:
    if isinstance(value, str):
        return value.format(**variables)
    if isinstance(value, list):
        return [_render_value(item, variables) for item in value]
    if isinstance(value, dict):
        return {key: _render_value(item, variables) for key, item in value.items()}
    return value


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
