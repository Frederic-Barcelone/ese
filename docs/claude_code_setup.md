# Claude Code -- Project Setup & Configuration

> **Date**: February 2026
> **Pipeline version**: v0.8

How Claude Code is configured and used in the ESE project for AI-assisted development, including permissions, plugins, development workflows, and persistent memory.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Project Structure](#2-project-structure)
3. [CLAUDE.md Manifest](#3-claudemd-manifest)
4. [Permission Configuration](#4-permission-configuration)
5. [Global Plugins](#5-global-plugins)
6. [Development Workflows](#6-development-workflows)
7. [Quality Rules & Verification](#7-quality-rules--verification)
8. [Persistent Memory](#8-persistent-memory)
9. [Project-Level Configuration](#9-project-level-configuration)
10. [Multi-Project Support](#10-multi-project-support)

---

## 1. Overview

Claude Code is Anthropic's CLI tool for AI-assisted software development. In the ESE project, it serves as the primary development assistant for:

- **Code implementation**: Writing new generators, validators, normalizers, and exporters
- **Debugging**: Systematic investigation of pipeline issues and test failures
- **Evaluation**: Running gold standard benchmarks and analyzing results
- **Documentation**: Generating and maintaining project documentation
- **Code review**: Checking code quality, identifying issues, and suggesting improvements

Claude Code reads the project's `CLAUDE.md` manifest to understand the codebase architecture, available commands, and coding conventions before making any changes.

---

## 2. Project Structure

### `.claude/` Directory

The project-level Claude Code configuration lives in `.claude/`:

```
.claude/
  settings.json          # Tracked in git -- shared permission rules
  settings.local.json    # Gitignored -- environment-specific permissions
```

### `CLAUDE.md` Manifest

The root-level `CLAUDE.md` file provides Claude Code with comprehensive project context:

```
CLAUDE.md                # Project manifest (checked into git)
pyproject.toml           # Python tooling config (mypy, ty)
```

### Memory Directory

Claude Code maintains persistent memory across sessions:

```
~/.claude/projects/-Users-frederictetard-Projects-ese/memory/
  MEMORY.md              # Cross-session context (auto-loaded)
```

---

## 3. CLAUDE.md Manifest

The `CLAUDE.md` file is the most important configuration -- it provides Claude Code with everything it needs to understand and work with the codebase.

### Contents

| Section | Purpose |
|---------|---------|
| **Project Overview** | Pipeline v0.8, 14+ entity types, core capabilities |
| **Commands** | How to run pipeline, tests, type checking, linting |
| **Environment Setup** | Python 3.12+, venv, API key configuration |
| **Architecture** | 6-layer pipeline diagram, layer philosophy table |
| **Directory Structure** | Full `corpus_metadata/` layout (A through Z layers) |
| **Operations** | Running the pipeline, log locations, output structure |
| **Configuration** | config.yaml sections, extraction presets, API settings |
| **LLM Cost Optimization** | Model tiers, 17 call types, pricing, call_type conventions |
| **Key Patterns** | Adding new entity types, generator interface, PASO heuristics |
| **External Dependencies** | Claude API, PubTator3, Unstructured.io, scispacy, etc. |
| **Lexicons** | 617K+ terms across 6 sources |
| **Claude Code Workflows** | Scale-based plugin usage, plugin reference table |
| **ESE-Specific Quality Rules** | Per-layer rules, verification checklist |

### How It Guides Behavior

When Claude Code starts a session, it reads `CLAUDE.md` to understand:

- **What commands to run**: `pytest K_tests/ -v`, `mypy .`, `ruff check .`
- **What patterns to follow**: Generator interface, model conventions, provenance tracking
- **What rules to enforce**: High recall for generators, high precision for validators
- **What to verify**: Mandatory pre-completion checklist (pytest + mypy + ruff)

---

## 4. Permission Configuration

### settings.json (Shared, Git-Tracked)

Defines 52 allowed bash commands available to Claude Code:

**Git operations (16 commands):**
- `git add`, `git commit`, `git push`, `git status`, `git diff`, `git log`
- `git rm`, `git mv`, `git fetch`, `git pull`, `git stash`
- `git ls-files`, `git check-ignore`, `git checkout`, `git reset`

**Python tooling (12 commands):**
- `python`, `python3`, `.venv/bin/python`, `.venv/bin/python3`
- `.venv/bin/pip`, `.venv/bin/pytest`, `.venv/bin/ruff`, `.venv/bin/mypy`
- `ruff`, `mypy`, `pytest`, `pip`, `pip3`

**File/system utilities (15 commands):**
- `ls`, `wc`, `sort`, `head`, `cat`, `echo`, `find`, `grep`
- `stat`, `sqlite3`, `open`, `kill`, `pkill`, `ps`, `lsof`

**Shell operations (4 commands):**
- `cd`, `source`, `tee`, `xargs`, `cut`

**Web access:**
- `WebSearch` -- allowed for research and documentation

### settings.local.json (Environment-Specific, Gitignored)

Defines environment-specific permissions including:

**WebFetch domain allowlist (24 domains):**
- Academic: `arxiv.org`, `aclanthology.org`, `link.springer.com`
- Biomedical: `www.ncbi.nlm.nih.gov`, `pmc.ncbi.nlm.nih.gov`, `ftp.ncbi.nlm.nih.gov`, `pubmed.ncbi.nlm.nih.gov`
- Development: `github.com`, `api.github.com`, `raw.githubusercontent.com`, `pypi.org`, `docs.astral.sh`
- Domain-specific: `www.genenames.org`, `docling-project.github.io`, `docs.unstructured.io`
- Data: `data.csiro.au`, `doi.org`, `www.sciencedirect.com`
- AI/ML: `www.confident-ai.com`, `langfuse.com`, `madewithml.com`

**Full-path Python commands:**
- `/Users/.../ese/.venv/bin/python -m pytest`, `mypy`, `ruff check`
- `PYTHONUNBUFFERED=1` prefix for real-time output
- Direct venv Python execution for pipeline runs

**Git operations with full paths:**
- `git -C /Users/.../ese log`, `git -C /Users/.../ese show`

---

## 5. Global Plugins

Five plugins are enabled globally for development workflows:

| Plugin | Purpose | Key Skills |
|--------|---------|------------|
| **superpowers** | Development workflow orchestration | `/brainstorming`, `/writing-plans`, `/systematic-debugging`, `/verification-before-completion`, `/test-driven-development` |
| **pyright-lsp** | Python type checking (Language Server Protocol) | Real-time type error detection |
| **code-review** | Code review against project guidelines | Style, patterns, and CLAUDE.md compliance |
| **pr-review-toolkit** | Pull request review automation | Code review, silent failure hunting, test analysis, type design |
| **code-simplifier** | Code simplification and cleanup | Reduce complexity while preserving functionality |

### superpowers Plugin Skills

| Skill | When to Use |
|-------|-------------|
| `/brainstorming` | Before any creative work -- explores intent, requirements, design |
| `/writing-plans` | Multi-step tasks -- creates structured implementation plans |
| `/test-driven-development` | Before implementation -- writes tests first |
| `/systematic-debugging` | Bug investigation -- systematic root cause analysis |
| `/verification-before-completion` | Before claiming done -- runs pytest, mypy, ruff |
| `/dispatching-parallel-agents` | 2+ independent tasks -- parallel execution |
| `/executing-plans` | Implementation from plan -- review checkpoints |
| `/finishing-a-development-branch` | After implementation -- merge/PR/cleanup decision |

---

## 6. Development Workflows

### Scale-Based Approach

Plugin usage scales with task size:

**Small changes** (config fix, doc update, 1-3 line fix):
- Just do the work directly
- Run verification commands before claiming done
- No plugins needed

**Medium changes** (bug fix, add method, modify behavior):
- `/systematic-debugging` for bugs
- `/verification-before-completion` before claiming done
- Optional: `/code-simplifier` for cleanup

**Large changes** (new entity type, multi-file refactor, new pipeline stage):
1. `/brainstorming` -- Explore requirements and design options
2. `/writing-plans` -- Create structured implementation plan
3. `/test-driven-development` -- Write tests first
4. Implement the plan
5. `/code-simplifier` -- Clean up the implementation
6. `/verification-before-completion` -- Full verification before completion

### ESE-Specific Plugin Notes

| Plugin | ESE-Specific Usage |
|--------|-------------------|
| `/brainstorming` | Explore recall vs precision tradeoffs for new extraction strategies |
| `/writing-plans` | Plan across all layers (A_core -> C_generators -> D_validation -> E_normalization -> J_export) |
| `/test-driven-development` | Write tests for generators, validators, and normalizers first |
| `/systematic-debugging` | Check each pipeline layer systematically |
| `/code-simplifier` | Preserve layer separation during cleanup |
| `/verification-before-completion` | Always run pytest, mypy, ruff |

---

## 7. Quality Rules & Verification

### Layer-Specific Rules

Each pipeline layer has specific quality expectations:

**Generators (C_generators/):**
- Optimize for high recall -- accept false positives
- Use FlashText for lexicon matching (not regex for large vocabularies)
- Every generator must implement `BaseCandidateGenerator` or `BaseExtractor` interface

**Validators (D_validation/):**
- Optimize for high precision -- filter aggressively
- LLM prompts must be versioned in the prompt registry
- Never auto-approve without an explicit PASO rule

**Normalizers (E_normalization/):**
- Map to standard ontologies (MONDO, RxNorm, HGNC)
- Handle API failures gracefully (PubTator, NCT)
- Deduplicate by canonical ID, not string matching

**Models (A_core/):**
- Strong Pydantic types with validators
- Provenance tracking on all extracted entities
- No optional fields without default values

### Mandatory Verification Checklist

Before marking any task complete, these three checks must pass:

```bash
# All tests pass (1,474 tests, ~1 second)
cd corpus_metadata && python -m pytest K_tests/ -v

# Type checking passes
cd corpus_metadata && python -m mypy .

# Linting passes
cd corpus_metadata && python -m ruff check .
```

No exceptions. All three must pass before any change is considered complete.

---

## 8. Persistent Memory

### MEMORY.md

Claude Code maintains a persistent `MEMORY.md` file that survives across sessions:

**Location**: `~/.claude/projects/-Users-frederictetard-Projects-ese/memory/MEMORY.md`

**Content** (auto-loaded into system prompt each session):
- **Key paths**: Python venv, working directory, datasource locations, gold data
- **F03 Evaluation Runner**: Config location, default settings, exit code behavior
- **NLP4RARE Gold Standard**: Annotation types, gold generation script, evaluation results
- **Disease Detection (C06)**: FlashText usage, apostrophe normalization fix, lexicon sources
- **Disease FP Filter (C24)**: Filter lists, confidence thresholds, adjustment floor
- **NLP4RARE Evaluation Progress**: Full benchmark results with TP/FP/FN/P/R/F1
- **BioCreative II GM Gold Standard**: Gene benchmark setup, results, schema mismatch analysis
- **CADEC Drug Gold Standard**: Drug benchmark setup, improvement trajectory, all fixes applied

**Guidelines:**
- Kept concise (lines after 200 are truncated)
- Separate topic files for detailed notes (e.g., `debugging.md`, `patterns.md`)
- Updated when common mistakes or new patterns are discovered
- Organized semantically by topic, not chronologically

### Cross-Session Context

The memory system enables Claude Code to:

- Remember benchmark results across sessions
- Avoid repeating known mistakes
- Build on previous debugging discoveries
- Maintain awareness of project conventions

---

## 9. Project-Level Configuration

### pyproject.toml

Python tooling configuration for mypy and ty:

```toml
[project]
name = "ese"
requires-python = ">=3.12"

[tool.mypy]
python_version = "3.12"
files = ["corpus_metadata"]
ignore_missing_imports = true
strict_optional = true
warn_unused_ignores = true
check_untyped_defs = true

[tool.ty.environment]
python-version = "3.12"
root = ["corpus_metadata"]
```

**Key settings:**
- Python 3.12+ required
- mypy strict optional checking enabled
- Missing import stubs ignored (third-party libraries)
- ty (Astral's fast type checker) configured as alternative

### Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The venv at `.venv/` provides isolated dependencies. Claude Code uses the full venv path for all Python commands: `.venv/bin/python`, `.venv/bin/pytest`, etc.

---

## 10. Multi-Project Support

### Main ESE Project

The primary project at `/Users/frederictetard/Projects/ese/`:

- Full pipeline codebase in `corpus_metadata/`
- Gold standard data in `gold_data/`
- Output datasources in `output_datasources/` (gitignored)
- Documentation in `docs/`
- Configuration in `.claude/settings.json` and `.claude/settings.local.json`

### CTIS Subproject

A separate subproject with its own Claude Code configuration:

- Located within the ESE repository
- Has its own `settings.local.json` for environment-specific permissions
- Shares the main project's `CLAUDE.md` and `settings.json`
- Separate concerns: clinical trial information system integration
