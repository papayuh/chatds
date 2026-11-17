# DSi SD storage selection

## Evidence and limits

At investigation time, public main and v0.2.0 were 8074642 (remote checked
read-only); no open/closed GitHub issues or newer pull requests fixed storage.
In that baseline, product `main.c` hardcoded `fat:/chatds/`, including independent
session/UI paths. `cds_config_read()` returned -1 on open failure, but main
called every failure `invalid run.txt` and then published failure results to the
same inaccessible directory. Running the baseline host binary with missing
run/output directories reproduced exactly
`invalid run.txt` and `result publication failed`. This proves the diagnostic
chain, **not** a DSi hardware diagnosis. Missing extraction, malformed config,
permissions, and launcher DS-mode restrictions remain alternative causes.

Release assets, ROM SHA256 and DS/DSpico evidence are recorded in the
[v0.2.0 release notes](../RELEASE-v0.2.0.md); that release does not claim DSi
support. During investigation, `ndstool -i` and header inspection showed unit
code 0x02 and flags at 0x1BF = 0x01 in both the release and candidate ROM; there
was no obvious DS-only header/build switch to change.
This still cannot prove TWiLight's actual launch mode or SD permissions.

Reddit plain GET returned HTTP 200 with an automatic challenge instead of thread
content; no bypass attempted. Browser connector was unavailable. The supplied
report says internal DSi SD + TWiLight Menu, but version, exact directory tree,
launcher version, DSi/DS mode and SD-access settings cannot be independently
confirmed. Ask for those before calling this confirmed.

Sources:
- https://github.com/papayuh/chatds/commit/8074642ddbc89ea3645744aa51616969589a3dfa
- https://blocksds.skylyrac.net/docs/guides/filesystem/
- https://github.com/blocksds/libnds/blob/master/source/arm9/libc/filesystem_setup.c
- Installed BlocksDS `libs/libnds/include/fat.h`
- https://www.reddit.com/r/NDSHacks/comments/1wyewew/comment/19sp7aqdaaeww/

## Initialization and design

Current BlocksDS implements the libfat-compatible API in libnds/FatFs.
`fatInitDefault()` mounts storage and chooses a default based on launcher argv
and execution mode. DSi mode tries internal SD and DLDI; required device follows
launch location. DS mode requires DLDI. Initialization is one-shot; repeated
calls do not retry. `fatGetDefaultDrive()` returns a borrowed root (`fat:/`,
`sd:/`, possibly `nand:/`), with no argv defaulting to SD in DSi mode and DLDI
in DS mode. `fatGetDefaultCwd()` allocates the ROM's directory: do not use it,
because assets must remain at card-root `/chatds/` even for nested ROM launches.
The header's initialization description is less detailed than current source;
rely on init's return value, not a claim that all devices are mounted.

After successful initialization, select one immutable application root through
one helper. Accept exactly `fat:/` => `fat:/chatds/` and `sd:/` => `sd:/chatds/`.
Reject NULL, unknown, NAND/NAND2/NitroFS before any application write. Do not
change mount code, enable NAND writes, create folders, probe write permissions
at boot, or choose a drive based on whether a config happens to exist. **No
cross-drive fallback**: missing/malformed input, read-only media or later write
failure stays on the selected card, so old files on another card cannot hide
errors or split reads and writes. Existing KB4-to-KB3 optional fallback remains
on that same root. DS-mode launches cannot gain internal SD access through a
path fix; launcher may need DSi mode.

## Path audit / scope

Route model.bin, tok.bin, kb.bin, kb4.bin, run.txt, round3-{act,op,tag}.cdsr,
kernel.txt, debug.log, ids/ctx/calc/heap/out.txt (+ .tmp), session.txt,
session-results.tsv, ui-script.txt, frames.txt, ui-result.txt and *.raw/*.txt
captures through the selected root. Check every constructed path, including
host asset overrides and temporary suffix space; reject overflow, never silently
truncate. Keep host CLI explicit input/output paths and environment override.
The separate `clean/ui/ds` diagnostic ROM uses `/chatds-ui/`; apply the same
selection policy there. `tools/asmkern/probe.c` uses root-relative files only in
a separate diagnostic ROM, not the shipped product; leave it unchanged.

## Diagnostics / failure modes

Distinguish open/read/close failure from config syntax/range errors. Include
actual run path and errno for I/O errors. Preserve original error first; a
secondary publication error reports its failing path and errno and does not
replace the original. Publication must return failure on open/write/flush/
close/rename/path-bound errors. Root/init rejection must not try publication.
Optional session/UI-script absence is still normal; other I/O failures should
be visible. Do not change the config grammar or output trailer contract.

## Tests and acceptance

- Behavioral host tests of actual selection/path helpers: flashcart, DSi,
  missing/unknown/NAND drives, exact bounds, no truncation and no fallback.
- Config tests: valid, absent, malformed, read failure. Product tests distinguish
  unreadable run.txt from malformed content, including simultaneous failed
  publication with original diagnostic retained.
- Publication tests: absent output directory, blocked temp file, rename failure,
  length limits; filename and errno in diagnostics. Audit all application I/O.
- Run host suite, ASan/UBSan engine tests and BlocksDS product/UI ROM builds.
- Run existing isolated headless melonDS DLDI harness against built product;
  its current harness has no DSi SD mode. Emulator evidence is not hardware
  compatibility. Record exact results and any unavailable tests in task report.
- Real DSi/TWiLight: record release/hash, extracted root tree, menu version,
  per-ROM DSi/DS mode and SD settings; launch from root and nested ROM folder,
  verify console selects sd:/chatds/, calculator/fact/learned/story turns,
  output/log files on SD only, missing and malformed config, unwritable output,
  and no fallback when a flashcart has a stale kit. Back up card before testing.
- Real DS Lite/DLDI: record cart/firmware/DLDI and ROM hash, same root assets;
  launch root/nested ROM and exercise UI/cancel/fact/learned/story turns,
  publication/overwrite and missing/malformed config. Selected root must stay
  fat:/chatds/. Compare deterministic answers/output metadata with baseline.

Accept when all scoped paths share one safely selected root, flashcart behavior
and on-disk layout stay intact, bounded paths fail closed, diagnostics distinguish
failures, host/build gates pass, and emulator/hardware limitations are explicit.
**DSi support remains unverified on real hardware until the above DSi run.**
No version bump, release publication or reporter reply is part of this change.

## Implementation notes

Policy and checked joins live in `clean/include/chatds_storage.h`; product and
standalone UI ROM use it. Input I/O errors preserve errno; the atomic publisher
reports open/write/remove/rename failures, and a failed debug log also forces
out.txt's status to FAIL. Product session/capture/kernel paths share the root.
Host CLI stays explicit, with oversized KB/learned paths rejected instead of
truncated. Host-only UI simulator paths are unrelated to launch-drive selection;
its CLI remains unchanged. No mount or NAND initialization code was added.

Behavioral policy tests use POSIX directories named `fat:`/`sd:`: these exercise
real helpers and host product I/O, **not** either hardware driver. The existing
headless melonDS runners exercise DLDI only. See task report for exact results,
artifact SHA256 and hardware plans; no DSi SD emulator/hardware claim follows
from a successful DS-mode DLDI boot.
