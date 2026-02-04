# Claude Code Configuration Reflection

Analysis of CLAUDE.md, `.claude/settings.local.json`, and session history to identify improvements for Claude Code's effectiveness on the ESE project.

---

## Issues Identified

### Issue 1: Verification Checklist References Non-Existent `--dry-run` Flag

**Current (CLAUDE.md line 324):**
```bash
python corpus_metadata/orchestrator.py --dry-run
```

**Problem:** `orchestrator.py` has no argparse, no `--dry-run` flag, and no CLI argument handling at all. This command would silently run the full pipeline on all PDFs or fail. Claude Code following this instruction could trigger an expensive full pipeline run with real API costs.

**Fix:** Remove the `--dry-run` command. The verification checklist should only include commands that actually exist.

---

### Issue 2: Commands Use Wrong Python Path

**Current (CLAUDE.md):**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Problem:** The project has two venvs (`venv/` and `.venv/`), and the settings.local.json permissions reference both paths extensively. The CLAUDE.md uses generic `python` which may resolve to the system Python. Commands should use the explicit venv path.

**Fix:** Reference the explicit venv path in all commands, and standardize on `.venv/` (the more conventional name).

---

### Issue 3: Extraction Presets List Is Incomplete

**Current (CLAUDE.md line 88-89):**
```yaml
preset: "standard"  # Options: drugs_only, diseases_only, abbreviations_only,
                    # feasibility_only, entities_only, all, minimal
```

**Problem:** The actual config.yaml defines 13 presets: `standard`, `all`, `minimal`, `drugs_only`, `diseases_only`, `genes_only`, `abbreviations_only`, `feasibility_only`, `entities_only`, `clinical_entities`, `metadata_only`, `images_only`, `tables_only`. CLAUDE.md lists only 7, missing 6 presets.

**Fix:** List all 13 presets.

---

### Issue 4: Core Capabilities Missing Entity Types

**Current (CLAUDE.md lines 10-18):** Lists 9 entity types.

**Problem:** Missing `Recommendations` (clinical guideline recommendations) and `Care Pathways` (patient journey mapping) which are fully implemented entity types with their own extractors, models, and exporters.

**Fix:** Add the missing entity types to the capabilities list.

---

### Issue 5: Architecture Diagram Is Incomplete

**Current:**
```
PDF → B_parsing → C_generators → D_validation → E_normalization → F_evaluation
```

**Problem:** This diagram omits layers I (extraction processors), J (export), H (pipeline orchestration), and Z (utilities). It also doesn't show the orchestrator or the actual data flow for entity types beyond abbreviations.

**Fix:** Show the complete data flow including all layers.

---

### Issue 6: settings.local.json Is Extremely Bloated (184 Entries)

**Problem:** The `.claude/settings.local.json` has 184 permission entries including:
- Entire git commit messages saved as individual permissions (lines 36, 43, 46, 65, 84, etc.)
- Shell loop constructs: `while`, `do`, `done`, `break`, `fi`, `then`, `else`
- Duplicate Python paths (`venv/` and `.venv/` variants)
- One-off file path references for specific gold data files
- Generic commands that should be in project settings (`git add`, `git commit`, `git push`)

**Fix:** Create a clean `.claude/settings.json` (project-level, committed) with broad patterns, and reset `.claude/settings.local.json` to minimal user-specific entries.

---

### Issue 7: No Custom Slash Commands

**Problem:** No `.claude/commands/` directory exists. The project would benefit from custom commands for common operations.

**Fix:** Create purpose-built slash commands for pipeline operations.

---

### Issue 8: Missing Operational Context

**Problem:** CLAUDE.md doesn't mention:
- The `corpus_log/` directory where all logs and `usage_stats.db` go
- How to run on a single PDF (config-based, not CLI args)
- Output goes alongside PDFs (e.g., `Pdfs/document_name/`)
- The pipeline version (`0.8`)

**Fix:** Add an Operations section.

---

### Issue 9: Plugin Workflow Section Is Over-Prescriptive

**Problem:** The "Mandatory Plugin Usage" section (lines 274-283) mandates 6 plugins for virtually every task. In practice, for small changes (config fix, doc update, single-line bug fix), invoking `/brainstorming` → `/writing-plans` → `/test-driven-development` → `/code-simplifier` → `/verification-before-completion` adds significant overhead without value. The chat history shows these plugins are rarely all invoked together.

**Fix:** Make plugins context-dependent rather than mandatory. Reserve the full workflow for substantial changes (new entity types, major refactors). Small fixes should only require `/verification-before-completion`.

---

### Issue 10: API Configuration Section Is Stale

**Current (CLAUDE.md lines 103-109):**
```yaml
api:
  claude:
    validation:
      model: "claude-sonnet-4-20250514"
```

**Problem:** This `validation.model` key is the legacy model selection. The pipeline now uses `model_tiers` for all model routing. Showing the old key alone is misleading.

**Fix:** Show both the default model and model_tiers together, with the note that model_tiers takes precedence.

---

## Proposed Changes

### 1. Updated CLAUDE.md

Key changes:
- Added pipeline version, care pathways, recommendations to capabilities
- Fixed all 13 presets listed
- Removed non-existent `--dry-run` command
- Added Operations section (logs, single PDF, output structure)
- Scaled back plugin mandates to be proportional to task size
- Fixed architecture diagram
- Added correct venv paths

### 2. New `.claude/settings.json` (Project-Level)

Clean, broad permissions that any developer would need:
```json
{
  "permissions": {
    "allow": [
      "Bash(git add:*)",
      "Bash(git commit:*)",
      "Bash(git push:*)",
      "Bash(git status:*)",
      "Bash(git diff:*)",
      "Bash(git log:*)",
      "Bash(git rm:*)",
      "Bash(git mv:*)",
      "Bash(git fetch:*)",
      "Bash(git pull:*)",
      "Bash(git stash:*)",
      "Bash(git ls-files:*)",
      "Bash(git check-ignore:*)",
      "Bash(git checkout:*)",
      "Bash(git reset:*)",
      "Bash(python:*)",
      "Bash(python3:*)",
      "Bash(.venv/bin/python:*)",
      "Bash(.venv/bin/python3:*)",
      "Bash(.venv/bin/pip:*)",
      "Bash(.venv/bin/pytest:*)",
      "Bash(.venv/bin/ruff:*)",
      "Bash(.venv/bin/mypy:*)",
      "Bash(ruff:*)",
      "Bash(mypy:*)",
      "Bash(pytest:*)",
      "Bash(pip:*)",
      "Bash(ls:*)",
      "Bash(wc:*)",
      "Bash(sort:*)",
      "Bash(head:*)",
      "Bash(cat:*)",
      "Bash(echo:*)",
      "Bash(find:*)",
      "Bash(grep:*)",
      "Bash(stat:*)",
      "Bash(sqlite3:*)",
      "Bash(open:*)",
      "Bash(kill:*)",
      "Bash(pkill:*)",
      "Bash(ps:*)",
      "Bash(lsof:*)",
      "Bash(cd:*)",
      "Bash(source:*)",
      "Bash(tee:*)",
      "Bash(xargs:*)",
      "Bash(cut:*)",
      "WebSearch"
    ]
  }
}
```

### 3. Reset `.claude/settings.local.json`

Reduce from 184 entries to only user-specific items (API-specific WebFetch domains, specific venv paths):
```json
{
  "permissions": {
    "allow": [
      "WebFetch(domain:arxiv.org)",
      "WebFetch(domain:docs.unstructured.io)",
      "WebFetch(domain:github.com)",
      "WebFetch(domain:api.github.com)",
      "WebFetch(domain:docs.astral.sh)",
      "WebFetch(domain:pypi.org)",
      "WebFetch(domain:www.genenames.org)",
      "WebFetch(domain:raw.githubusercontent.com)",
      "WebFetch(domain:docling-project.github.io)"
    ]
  }
}
```

---

## Implementation Status

- [ ] Update CLAUDE.md with all fixes above
- [ ] Create `.claude/settings.json` with clean project-level permissions
- [ ] Reset `.claude/settings.local.json` to minimal user-specific entries
- [ ] Remove EXTRACTOR.MD (redundant with docs/)
- [ ] Fix pipeline version discrepancy (config.yaml 0.7 → 0.8)
