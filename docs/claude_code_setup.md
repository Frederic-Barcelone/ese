# Claude Code -- Project Setup & Configuration

> **Pipeline version**: v0.8

---

## 1. Overview

Claude Code (Anthropic CLI) reads `CLAUDE.md` to understand the ESE codebase for code implementation, debugging, evaluation, and review.

---

## 2. Project Structure

```
.claude/
  settings.json          # Git-tracked -- shared permission rules
  settings.local.json    # Gitignored -- environment-specific permissions

CLAUDE.md                # Project manifest (checked into git)
pyproject.toml           # Python tooling config (mypy, ty)

~/.claude/projects/-Users-frederictetard-Projects-ese/memory/
  MEMORY.md              # Cross-session persistent context
```

---

## 3. CLAUDE.md Manifest

Covers: project overview, commands, architecture (6-layer pipeline), config.yaml, LLM cost optimization (17 call types, 2 tiers), key patterns (adding entities, generator interface, PASO), external dependencies, lexicons (617K+ terms), and quality rules. Enforces mandatory verification checklist (pytest, mypy, ruff).

---

## 4. Permission Configuration

### settings.json (Shared, Git-Tracked)

52 allowed bash commands across git (16), Python (12), file/system (15), shell (5), and WebSearch.

### settings.local.json (Gitignored)

WebFetch allowlist (24 domains), full-path Python/git commands with venv paths.

---

## 5. Global Plugins

| Plugin | Purpose |
|--------|---------|
| **superpowers** | `/brainstorming`, `/writing-plans`, `/systematic-debugging`, `/verification-before-completion`, `/test-driven-development`, `/dispatching-parallel-agents`, `/executing-plans`, `/finishing-a-development-branch` |
| **pyright-lsp** | Real-time type error detection |
| **code-review** | Style/pattern review against CLAUDE.md |
| **pr-review-toolkit** | PR review, silent failure hunting, test analysis |
| **code-simplifier** | Reduce complexity, preserve functionality |

---

## 6. Development Workflows

- **Small** (1-3 lines): Just do it. Verify before done.
- **Medium** (bug fix): `/systematic-debugging`, then `/verification-before-completion`.
- **Large** (new entity type): `/brainstorming` -> `/writing-plans` -> `/test-driven-development` -> implement -> `/code-simplifier` -> `/verification-before-completion`

---

## 7. Quality Rules & Verification

### Layer Rules

- **Generators (C)**: High recall, accept FPs. FlashText for lexicons. Implement `BaseCandidateGenerator`/`BaseExtractor`.
- **Validators (D)**: High precision. Versioned prompts. No auto-approve without PASO rule.
- **Normalizers (E)**: Standard ontologies (MONDO, RxNorm, HGNC). Graceful API failure handling. Dedup by canonical ID.
- **Models (A)**: Strong Pydantic types. Provenance tracking. No optional fields without defaults.

### Mandatory Verification

```bash
cd corpus_metadata && python -m pytest K_tests/ -v
cd corpus_metadata && python -m mypy .
cd corpus_metadata && python -m ruff check .
```

No exceptions. All three must pass.

---

## 8. Persistent Memory

**Location**: `~/.claude/projects/-Users-frederictetard-Projects-ese/memory/MEMORY.md`

Auto-loaded each session. Contains: key paths, F03 config, gold standard details, FP filter thresholds, benchmark results, bug fixes. Organized by topic.

---

## 9. Project Configuration

### pyproject.toml

```toml
[tool.mypy]
python_version = "3.12"
files = ["corpus_metadata"]
ignore_missing_imports = true
strict_optional = true
warn_unused_ignores = true
check_untyped_defs = true
```

Python 3.12+ required. Claude Code uses full venv paths (`.venv/bin/python`, `.venv/bin/pytest`).

---

## 10. Multi-Project Support

- **Main ESE**: `/Users/frederictetard/Projects/ese/` -- pipeline in `corpus_metadata/`, gold data in `gold_data/`, docs in `docs/`
- **CTIS subproject**: Separate `settings.local.json`, shares main `CLAUDE.md` and `settings.json`
