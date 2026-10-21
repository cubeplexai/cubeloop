# Runbook: Cutting a docs version snapshot

When CubePi cuts a new release `X.Y`, the Docusaurus site needs a frozen
snapshot of the docs under `website/versioned_docs/version-X.Y/` (and the
zh-Hans mirror) so future docs drift on `main` doesn't rewrite history for
readers landing on the released version. This runbook captures the order of
operations and the traps the 0.5 → 0.6 → 0.7 cuts surfaced.

## When to run

Run this **right before** tagging the release — after all feature work for
`X.Y` has landed on `main` (or your release-prep worktree) and `current/` is
in the exact state you want to ship. The cut produces a frozen copy; anything
you fix in `current/` after the cut won't reach the released version without
a separate edit.

## Prerequisites

- Working in the release-prep worktree (`.worktrees/YYYY-MM-DD-release-X.Y-...`).
- `pyproject.toml` already bumped to `X.Y.0`.
- `CHANGELOG.md` has the `## [X.Y.0]` section populated (the `/changelog`
  page reads from the repo root, so it's not part of the snapshot — but
  prose in versioned guides may reference it).
- `current/` (EN + zh-Hans) and the under-development code are in sync.
  Specifically, every new public API in `X.Y` should already have a guide
  page and the page should match the shipped behavior. Use `diff -r
  website/docs website/i18n/zh-Hans/docusaurus-plugin-content-docs/current`
  to spot stragglers.

## The cut, in order

All commands run from `website/`.

```bash
# 1. Regenerate API mdx from the Python source (gitignored in current/, but
#    the snapshot will commit them under versioned_docs/version-X.Y/api/).
pnpm apiref

# 2. Snapshot. This populates:
#    - website/versioned_docs/version-X.Y/
#    - website/i18n/zh-Hans/docusaurus-plugin-content-docs/version-X.Y/
#      and version-X.Y.json (so the locale stays paired)
#    - website/versioned_sidebars/version-X.Y-sidebars.json
#    - prepends "X.Y" to website/versions.json
pnpm docusaurus docs:version X.Y

# 3. Edit website/docusaurus.config.ts (see "Config edits" below).

# 4. Build with the new version flipped to latest.
pnpm build
```

## Config edits (`docusaurus.config.ts`)

In the `classicOptions.docs` block:

- Flip `lastVersion` to `'X.Y'`.
- In `versions`:
  - Add `'X.Y': { label: 'X.Y (latest)', path: '' }`.
  - Demote the previous `(latest)` entry to `{ label: 'N-1', path: 'N-1',
    noIndex: true }`. `noIndex: true` is **mandatory** — without it,
    Google reindexes the demoted docs as duplicate content of the new
    latest, costing the new version search rank.
- In `sitemap.ignorePatterns`, add `'/docs/N-1/**'` so the demoted version
  is dropped from `sitemap.xml`.

Don't touch the `current: { label: 'Next 🚧', path: 'next', banner:
'unreleased', noIndex: true }` entry — `current/` should always live at
`/docs/next/` with `noIndex` so unreleased changes don't pollute search.

## Gotchas (the meat)

These are the things that have actually bitten us during 0.5 → 0.8 cuts. Read
them before running the cut, not after the build fails.

### 1. Run `pnpm apiref` **before** `docusaurus docs:version`

`website/docs/api/cubeloop-*.mdx` is gitignored (only `index.mdx` is committed)
because it's regenerated from Python source on every build by `pnpm apiref`
(wired into `prebuild`/`prestart`). If you cut without running `apiref`
first, the snapshot copies an empty `api/` directory, the next build fails
on a broken link from `api/index.mdx`, and you have to hand-port the api
mdx files into `versioned_docs/version-X.Y/api/` after the fact. The 0.6
cut hit this — see commit `ee1056d` (`fix(docs): add generated API reference
to version-0.6 snapshot`).

### 2. The snapshot freezes `sidebars.ts` too — new pages must be wired before the cut

`docusaurus docs:version` writes the **current** sidebar into
`versioned_sidebars/version-X.Y-sidebars.json`. Any sidebar entry you add
to `sidebars.ts` after the cut shows up only on `current/` (Next 🚧), not
on `X.Y`. If you forgot a page, the fix is either (a) re-cut by deleting
`versioned_docs/version-X.Y/` and `versioned_sidebars/version-X.Y-sidebars.json`
and rerunning, or (b) hand-edit the versioned sidebar JSON.

**Pre-cut check:** every new guide page in `website/docs/` for this release
must already be listed in `sidebars.ts`. A page that is in the file tree but
not in the sidebar is silently unreachable from navigation on both `current/`
and the new versioned snapshot. The 0.8 cut discovered that `forking.md` was
missing from `sidebars.ts` entirely (the fork PR forgot to add it) — caught
only by the codex review after the snapshot was already committed, requiring a
hand-edit of both `sidebars.ts` and `versioned_sidebars/version-0.8-sidebars.json`.

### 3. `current/` and `version-X.Y/` must be byte-identical at cut time

Cutting at release time means a reader on `/docs/next/foo` and a reader on
`/docs/foo` should see the same thing the day of the release. Divergence
happens silently when:

- One locale was updated but the other wasn't.
- Old code samples weren't refreshed for the new API (the 0.7 cut had 13
  pages still importing the old `Model` symbol after the BoundModel
  redesign).
- A hero/landing card hardcodes a path to one specific provider page
  instead of using `custom-versionAwareDocLink` (the 0.7 hero linked to
  `/guides/providers/anthropic` instead of `/guides/providers/overview`,
  which then drifted between current and version-0.7).

After cutting, run `diff -r website/docs website/versioned_docs/version-X.Y`
and the zh-Hans equivalent. The only expected difference is `intro.mdx`
status blurb: `current/` has the "Next 🚧" copy, the snapshot has the
hardcoded Released copy (see gotcha #8 and #9).

### 4. zh-Hans parity is your problem

`docusaurus docs:version` does mirror the zh-Hans locale into
`i18n/zh-Hans/docusaurus-plugin-content-docs/version-X.Y/` and writes
`version-X.Y.json`, but only the *files that exist* in `current/` at cut
time. If a guide page was added to EN `website/docs/` but its zh-Hans
mirror wasn't written to `website/i18n/zh-Hans/.../current/`, the snapshot
permanently has an EN-only page. Verify before cutting:

```bash
diff <(cd website/docs && find . -name '*.md*' | sort) \
     <(cd website/i18n/zh-Hans/docusaurus-plugin-content-docs/current && find . -name '*.md*' | sort)
```

Any output is a missing translation. Write it first, then cut.

### 5. Version-aware links, not hardcoded paths

Anywhere a doc, landing page, or React component links to another doc, use
the `custom-versionAwareDocLink` navbar item type or relative markdown
links (`./other-page` resolves correctly across versions). Hardcoded
absolute paths like `/docs/guides/providers/anthropic` work on the version
they were written for but send `0.6` readers to the `0.7` page after the
cut. The 0.7 hero card hit this — the fix is to point cross-section links
at section indices (`overview` pages) rather than specific subpages.

### 6. Don't commit generated API mdx into `current/`

CI has a guard that fails the build if `website/docs/api/cubeloop-*.mdx` is
present in a commit (only `index.mdx` is allowed). The snapshot is allowed
to contain them because the path is different
(`versioned_docs/version-X.Y/api/`). If your local cut accidentally staged
`website/docs/api/cubeloop-*.mdx`, unstage them — they regenerate on every
build.

### 7. `versions.json` order matters

The cut prepends `X.Y` to `versions.json`. Don't manually reorder — the
order is `[latest, ..., oldest]` and Docusaurus uses it for the version
dropdown. If you intentionally drop an old version, delete its entry
from `versions.json` **and** its `versioned_docs/version-N/` directory,
sidebar JSON, and any locale mirrors, plus its entry from the `versions`
block in `docusaurus.config.ts`.

### 8. Status blurb in `intro.mdx`

Both `website/docs/intro.mdx` and the zh-Hans mirror typically carry a
"Status" line that should mention what's new in `X.Y`. The snapshot
preserves whatever was in `current/` at cut time, so write the Released
copy in `current/` **before** the cut and then revert `current/`'s blurb
to the Next 🚧 copy after the cut so unreleased work isn't advertised as
shipped.

### 9. Hardcode the version number in the snapshot's `intro.mdx`

The pre-cut "Released" copy of `intro.mdx` (see gotcha #8) must **not**
use `<PackageVersion />` in the versioned snapshot. `<PackageVersion />`
reads `pyproject.toml` at build time — after the next release bumps the
version, the frozen 0.8 snapshot would render "CubePi v0.9.0 is the latest
release" instead of remaining a 0.8 page.

Replace any `<PackageVersion />` in the snapshot's status blurb with a
hardcoded version string:

```mdx
<!-- ❌ wrong — reads pyproject.toml at build time -->
**CubePi v<PackageVersion /> is the latest release.**

<!-- ✅ correct — frozen in the snapshot -->
You are reading the **CubePi X.Y** release docs. This is the latest stable
documentation set for installing CubePi and building against the published
X.Y API.
```

The pattern from `versioned_docs/version-0.7/intro.mdx` is the canonical
reference. Apply the same fix to the zh-Hans snapshot
(`i18n/zh-Hans/.../version-X.Y/intro.mdx`). The `current/` copy may
continue to use `<PackageVersion />` — it always tracks the real latest.

### 10. Fix the zh-Hans `version-X.Y.json` label after the cut

`docusaurus docs:version` creates
`i18n/zh-Hans/docusaurus-plugin-content-docs/version-X.Y.json` by copying
the `current` locale metadata verbatim. That means `version.label` starts
as `"Next 🚧"` with a description of `"The label for version current"`.
Chinese readers will see the released X.Y entry in the version picker
labelled "Next 🚧", making the released and unreleased channels
indistinguishable.

After the snapshot, open the new `version-X.Y.json` and update:

```json
{
  "version.label": {
    "message": "X.Y（最新）",
    "description": "The label for version X.Y"
  },
  ...
}
```

Once this version is later demoted (at X+1.Y), change the message to just
`"X.Y"` (drop "（最新）") to match the EN config pattern.

### 11. Add the `[X.Y.0]` CHANGELOG reference-link and bump `[Unreleased]`

`CHANGELOG.md` follows Keep a Changelog with reference-style links — the
`## [X.Y.0] - YYYY-MM-DD` heading only renders as a link to the compare
view when there is a matching `[X.Y.0]: https://...compare/v(X.Y-1).0...vX.Y.0`
entry in the link block at the bottom of the file. The `[Unreleased]`
entry must also be re-pointed each release.

The 0.9 and 0.10 cuts both shipped with `## [X.Y.0]` rendering as plain
text because the link block wasn't bumped. The `/changelog` docs page
and the GitHub blob view both make the broken reference visually
obvious — readers expect the prior-version pattern.

When promoting `[Unreleased]` → `[X.Y.0] - YYYY-MM-DD`, also patch the
link block at the bottom of `CHANGELOG.md`:

```diff
-[Unreleased]: https://github.com/cubeplexai/cubeloop/compare/v(X.Y-1).0...HEAD
+[Unreleased]: https://github.com/cubeplexai/cubeloop/compare/vX.Y.0...HEAD
+[X.Y.0]: https://github.com/cubeplexai/cubeloop/compare/v(X.Y-1).0...vX.Y.0
 [(X.Y-1).0]: https://github.com/cubeplexai/cubeloop/compare/v(X.Y-2).0...v(X.Y-1).0
```

Verify before commit — every `## [N.M.0]` heading must have a matching
`[N.M.0]:` reference. Use a subset check, not a strict equality diff,
because reference IDs are also reused inline (e.g. `**[0.4.0]**` in the
"Earlier releases" prose):

```bash
# Lists any version heading missing a reference target. Empty output = clean.
comm -23 \
  <(grep -oE '^## \[[0-9]+\.[0-9]+\.[0-9]+\]' CHANGELOG.md | tr -d '# ' | sort -u) \
  <(grep -oE '^\[[0-9]+\.[0-9]+\.[0-9]+\]'    CHANGELOG.md | sort -u)
```

## Post-cut verification

After committing the cut, before opening the PR / tagging:

```bash
cd website
pnpm apiref               # regenerate (idempotent)
pnpm build                # must pass with no broken-link errors
pnpm test                 # vitest suite covers the version-aware nav
cd ..
# CHANGELOG link sanity (see gotcha #11). Empty output = every
# "## [N.M.0]" heading has a matching reference target.
comm -23 \
  <(grep -oE '^## \[[0-9]+\.[0-9]+\.[0-9]+\]' CHANGELOG.md | tr -d '# ' | sort -u) \
  <(grep -oE '^\[[0-9]+\.[0-9]+\.[0-9]+\]'    CHANGELOG.md | sort -u)
```

Then in the rendered site (`pnpm start` or the preview deploy):

- `/docs/` redirects to the new `X.Y` content with `X.Y (latest)` in the
  version dropdown.
- `/docs/next/` shows the same content with the `unreleased` banner.
- `/docs/N-1/` still serves the demoted version, but `<meta name="robots"
  content="noindex">` is present in its HTML head.
- `sitemap.xml` contains `/docs/...` paths but not `/docs/N-1/...` or
  `/docs/next/...`.

## After the cut

Push the annotated `vX.Y.Z` tag only after main CI and the docs deploy for
the merge commit have succeeded. `.github/workflows/publish.yml` runs on
that tag. It publishes both packages to TestPyPI and then PyPI, then creates
the GitHub Release.

The Release body is the matching `## [X.Y.Z] - YYYY-MM-DD` section of
`CHANGELOG.md`, followed by the pull requests merged since the previous
release. The workflow stops before publishing if that section is missing or
empty, or if `pyproject.toml` and `compat/cubepi/pyproject.toml` are not
both exactly `X.Y.Z`. Keep the changelog on the tagged commit. A rerun reads
that commit and replaces the Release body.

A Release created by hand still starts the same workflow, and the generated
body replaces whatever was typed in. The usual path is to push the tag and
let the workflow create the Release.
