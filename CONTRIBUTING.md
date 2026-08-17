# Contributing

## Commits (Commitizen)

```text
<type>(<scope>): <description>
```

Types we use: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`, `ci`.

```bash
cz commit          # or write the message by hand
cz bump            # after merges; 0.x, never auto-1.0.0
```

- `feat:` → 0.**y**.z
- `fix:` / `perf:` → 0.y.**z**
- `BREAKING CHANGE:` → 0.**y**.0 (because `major_version_zero = true`)
- Do **not** cut 1.0.0 until Phase 1 launches a real module and we say so.

## Branch / PR

- Work on `feat/…` or `fix/…` off `main`.
- PR to `main`. CI must be green.
- Do not add peft / qlora / axolotl / rust-ai-core / unsloth-rs as deps.

## Honesty

`bridge_ready()` stays `false` until issue #3 has a load + launch that
touches a device pointer and a numerical test. Empty `cuda`/`python`
features must not flip it.

## GPU work

This sandbox has no GPU. See [docs/GPU_HANDOFF.md](docs/GPU_HANDOFF.md)
for the Claude / RTX 5080 debug-artifact loop.
