<!-- Short project-specific instructions for GitHub Copilot / AI coding agents -->
# Vibe Eucher — Copilot Instructions

-Purpose: Give GitHub Copilot concise, actionable, repository-specific guidance so suggestions and generated code follow local conventions.

Project layout
- **Code:** `src/`
- **Tests:** `tests/`
- **Examples / scripts:** `examples/`, `scripts/`
- **Models / checkpoints (do not edit):** `models/`, `models/checkpoints/`

Enforcement & safety (SHALL / SHALL NOT)
- **No `sudo`:** never run `sudo`; all commands operate in userspace. See `.cursor/rules/safety/no-sudo.mdc`.
- **No destructive home deletes:** never use `rm -rf ~` or similar. See `.cursor/rules/safety/no-home-rm.mdc`.
- **Command-line safety:** avoid `!` in strings intended for `bash`. See `.cursor/rules/safety/command-line.mdc`.

Python & environment
- **pyproject.toml only:** use `pyproject.toml` for project metadata — do not add `setup.py`. See `.cursor/rules/python/pyproject-toml.mdc`.
- **Virtualenv required & named:** use a venv; name it `venv_$(basename $(pwd))` (in this repo it is `venv_vibe_eucher/`). Activate before running tests: `source venv_vibe_eucher/bin/activate`. See `.cursor/rules/python/venv-required.mdc` and `venv-naming.mdc`.
- **Makefile venv paths:** in Makefiles prefer full venv bin paths (e.g. `venv_vibe_eucher/bin/pytest`, `venv_vibe_eucher/bin/pip`). See `.cursor/rules/python/venv-paths.mdc`.
- **Python 3.10+ idioms:** target Python >= 3.10; prefer modern typing (`str | int`), dataclasses, `pathlib`, `match/case`, async where appropriate. See `.cursor/rules/python/python-version.mdc`.
- **Type checking & docs:** use mypy typing and NumPy-style docstrings; run documentation linting. See `.cursor/rules/python/typing.mdc` and `documentation-linting.mdc`.
- **Formatting:** use `ruff` to check & format `.py` and `.ipynb` files. See `.cursor/rules/python/ruff-check-format.mdc`.
- **Line length:** 120 chars. See `line-length.mdc`.

Makefile & build
- **Target dependencies:** Makefile targets that create files/folders SHOULD declare those targets so `make` orders correctly (example: venv creation target). See `makefile/target-dependencies.mdc`.
- **Variable escaping:** prefer `${VAR}` instead of `$VAR` in Makefiles. See `makefile/variable-escaping.mdc`.

Git & commits
- **User config for AI commits:** automated commits SHALL set git user to `$(whoami) | Cursor.sh | <model>`. See `git/user-config.mdc`.
- **Commit format:** strict format required (summary, blank, `- ` lines, blank, `-----`, technical attribution). See `git/commit-format.mdc`. Use the template in this file for AI-generated commits.
- **Upstream sync before commit:** fetch and merge upstream before committing; push after commit if remote exists. See `git/upstream-sync.mdc` and `git/push-requirement.mdc`.

Endpoints & testing
- **Prefer real endpoints:** tests and API code SHOULD prefer real endpoints and real authentication for happy-path testing; use mocks only for unavailable or error scenarios. See `endpoints/prefer-real.mdc`.
- **Authentication & retries:** on unauthenticated responses attempt real authentication; on auth failures exit tests with clear errors; network failures should use exponential backoff and log detailed errors. See `endpoints/authentication.mdc` and `endpoints/error-handling.mdc`.

Tools & logs
- **Mitmproxy logs:** do not treat mitmproxy logs as plain text; use `mitmdump` for analysis. See `tools/mitmproxy-logs.mdc`.
- **UV for package ops:** use `uv` for package transactions where applicable (except initial setup). See `python/uv-usage.mdc`.

Practical examples (copyable)
- Run tests (from repo root):
```bash
source venv_vibe_eucher/bin/activate
venv_vibe_eucher/bin/pytest -q
```
- Makefile venv target example:
```makefile
venv_vibe_eucher:
	python -m venv venv_vibe_eucher

test: venv_vibe_eucher
	venv_vibe_eucher/bin/pytest
```

Editing, PRs & traceability
- Keep changes small and focused; mirror the atomic nature of `.cursor/rules/` when creating PRs.
- Add or update tests in `tests/` for behavior changes; prefer exercising real endpoints where practical.
- Do NOT add or modify files under `models/` or `models/checkpoints/`.

Where to look for precise rules
- The authoritative source of project rules is the `.cursor/rules/` directory. When in doubt, reference the matching `.mdc` (for example `.cursor/rules/endpoints/prefer-real.mdc`) and include the `globs:` entry in your PR description so reviewers can validate the change.

If you need more detail
- Tell me which specific rules you want expanded with examples and I will integrate them (I can inline small code examples from files like `src/cli.py`, `src/game.py`, or relevant tests).

-- End of file

Inline examples and quick templates (by rule file)

- `.cursor/rules/git/commit-format.mdc` — Commit message template (use verbatim):

```text
Short summary of change

- Add feature X
- Fix bug Y

-----
Prompt: <brief prompt used>
Context: <one-line description of what the code does>

Technical details:
- Model: <name/version if applicable>
- IDE: Cursor
- Generation method: AI-assisted pair programming
- Code style: <language/style guide>
- Dependencies: <key deps>
```

- `.cursor/rules/git/upstream-sync.mdc` — Upstream sync steps (run before commit):

```bash
git fetch upstream && git fetch origin
UPSTREAM_BRANCH=$(git symbolic-ref refs/remotes/upstream/HEAD 2>/dev/null | sed 's@^refs/remotes/upstream/@@' || echo main)
git merge upstream/${UPSTREAM_BRANCH}
# resolve conflicts if any, then commit and push
```

- `.cursor/rules/git/user-config.mdc` — Set git user for automated commits:

```bash
git config user.name "$(whoami) | Cursor.sh | <model>"
git config user.email "$(whoami)@local"
```

- `.cursor/rules/makefile/target-dependencies.mdc` and `variable-escaping.mdc` — Makefile examples:

```makefile
venv_vibe_eucher:
	python -m venv venv_vibe_eucher

test: venv_vibe_eucher
	venv_vibe_eucher/bin/pytest

# Use full escapes for variables
PROJECT := $(notdir $(CURDIR))
VENV := venv_${PROJECT}

```

- `.cursor/rules/python/venv-naming.mdc` & `venv-required.mdc` — venv commands:

```bash
# create (if missing) and activate venv
python -m venv venv_vibe_eucher
source venv_vibe_eucher/bin/activate
```

- `.cursor/rules/python/pyproject-toml.mdc` — Project configuration: always prefer `pyproject.toml`; do not add `setup.py`.

- `.cursor/rules/python/venv-paths.mdc` — Use full venv bin paths in Makefiles and scripts, e.g. `venv_vibe_eucher/bin/pip`.

- `.cursor/rules/python/python-version.mdc` — Idiomatic modern Python examples:

```py
from dataclasses import dataclass
from pathlib import Path

def parse(value: str | int) -> str:
	match value:
		case int():
			return f"{value}"
		case str():
			return value

@dataclass
class Config:
	root: Path
```

- `.cursor/rules/python/typing.mdc` — Use mypy-style typing everywhere. Example:

```py
def score(player: str, points: int) -> float:
	return points / 10.0
```

- `.cursor/rules/python/ruff-check-format.mdc` — Format and check:

```bash
venv_vibe_eucher/bin/ruff check .
venv_vibe_eucher/bin/ruff format .
```

- `.cursor/rules/python/line-length.mdc` — Line length: 120 chars.

- `.cursor/rules/python/documentation.mdc` & `documentation-linting.mdc` — Use NumPy-style docstrings and run doc linting (e.g., pydocstyle or ruff plugins).

- `.cursor/rules/python/uv-usage.mdc` — Use `uv` for package operations when available (example):

```bash
# install a package quickly using uv
uv add requests
```

- `.cursor/rules/endpoints/prefer-real.mdc` — Test pattern: prefer real endpoints. Example test skeleton:

```py
def test_remote_auth_and_game_flow():
	# attempt real authentication (use env var for creds)
	token = real_auth_from_env()
	assert token
	resp = requests.get(api_url('/game/state'), headers={"Authorization": f"Bearer {token}"})
	assert resp.status_code == 200
```

- `.cursor/rules/endpoints/authentication.mdc` & `error-handling.mdc` — Authentication and retry pattern:

```py
def call_with_reauth(client, url, max_retries=3):
	for attempt in range(max_retries):
		r = client.get(url)
		if r.status_code == 401:
			if not reauthenticate():
				raise RuntimeError("Auth failed; aborting test")
			continue
		if r.status_code >= 500:
			backoff(attempt)
			continue
		return r
	raise RuntimeError("Retries exhausted")
```

- `.cursor/rules/tools/mitmproxy-logs.mdc` — Analyze mitmproxy logs with `mitmdump`:

```bash
mitmdump -r capturefile --flow-detail 1
```

- `.cursor/rules/safety/*` — Warnings (do not run):

```text
# NEVER run these (examples):
rm -rf ~
sudo some-command-that-modifies-system
```

- `.cursor/rules/git/push-requirement.mdc` — Push after commit if remote exists. Example:

```bash
git commit -m "..."
if git remote | grep -q .; then git push; fi
```

If you'd like I can now:
- Expand any selected rule into longer examples placed into `docs/` or inline code examples in `examples/`.
- Create a commit for this update following the required commit format and upstream-sync steps.

Which would you like next? 
