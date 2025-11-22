# Cursor Rules

Cursor rules organized into atomic, focused, and actionable units.

## Organization

Rules are organized into category folders, with each rule file containing a single, atomic requirement:

```
/projects/cursor_rules/
├── README.md
├── git/
│   ├── user-config.mdc          # Git user configuration
│   ├── commit-format.mdc         # Commit message format
│   ├── push-requirement.mdc      # Push after commit
│   └── upstream-sync.mdc         # Upstream sync workflow
├── makefile/
│   ├── variable-escaping.mdc     # Variable escaping
│   ├── target-dependencies.mdc   # Target dependencies
│   └── phony-venv.mdc            # .PHONY exclusion
├── python/
│   ├── venv-required.mdc         # Virtual environment requirement
│   ├── venv-naming.mdc            # Venv naming convention
│   ├── venv-paths.mdc             # Venv paths in Makefiles
│   ├── typing.mdc                 # Mypy typing
│   ├── documentation.mdc          # NumPy docstrings
│   ├── pyproject-toml.mdc          # pyproject.toml requirement
│   └── python-version.mdc         # Python 3.10+ and modern practices
├── safety/
│   ├── no-home-rm.mdc             # Home directory protection
│   ├── no-sudo.mdc                # Sudo prohibition
│   └── command-line.mdc           # Command line safety
├── tools/
│   └── mitmproxy-logs.mdc         # Mitmproxy log handling
├── endpoints/
│   ├── prefer-real.mdc            # Real endpoints over mocks
│   ├── authentication.mdc         # Authentication handling
│   └── error-handling.mdc         # Error handling
├── requirements/
│   ├── hierarchy.mdc              # Document hierarchy
│   ├── workflow.mdc               # Workflow commands
│   └── practices.mdc              # Best practices
└── workflow.mdc                    # General workflow
```

## Rule Quality Standards

All rules:
- Are focused, actionable, and scoped
- Are under 500 lines
- Are composable (split large rules into multiple rules)
- Provide concrete examples or referenced files
- Avoid vague guidance (written like clear internal docs)
- Are reusable when repeating prompts in chat

## File-Specific Application

Rules use globs patterns in frontmatter to apply only to relevant files:
- Python rules apply to `*.py` files
- Makefile rules apply to `Makefile` and `makefile` files
- Safety and git rules apply to all files (universal requirements)
- Tool rules apply to specific file types (e.g., `*.log` for mitmproxy)
- Endpoint rules apply to test and API files
- Requirements rules apply to requirements documentation files

This ensures rules like NumPy documentation requirements only apply to Python files, not MATLAB files.

## Requirements Language

All rules use requirements language:
- **SHALL** - Mandatory requirement
- **SHALL NOT** - Prohibited action
- **MUST** - Required for correctness
- **SHOULD** - Recommended but not mandatory