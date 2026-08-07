# Publishing the Wiki

The Markdown under [`wiki/`](https://github.com/eantcal/nunn/tree/main/wiki) is the reviewable source copy. GitHub renders the public wiki from a separate repository:

```text
https://github.com/eantcal/nunn.wiki.git
```

A documentation change is complete only when the source repository and wiki repository contain the same page and asset content.

## Editorial contract

Topic pages follow one consistent progression:

1. state what problem the model solves;
2. show the relevant equation or data shape;
3. include a short snippet checked against the current public API or example;
4. link header, implementation, test, and runnable example;
5. state constraints and diagnostics;
6. end with a “Keep reading” link.

Additional conventions:

- use English for the English-edition wiki;
- treat checked-in code and tests as authoritative;
- link source files on the `main` branch;
- label shortened snippets as reduced or source-backed and do not invent helper APIs;
- keep code blocks small enough to explain one operation;
- prefer stable file links over fragile line-number anchors;
- keep asset filenames descriptive and stable;
- do not duplicate the whole README or book chapter.

## Validate the source copy

From the repository root:

```powershell
.\scripts\check-wiki.ps1
git diff --check
```

The checker verifies internal page links, asset references, repository source links, page headings, and the absence of stale `blob/master` URLs.

For code snippets, also build the linked examples and run the test suite:

```sh
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

## Review the two repositories before publishing

```powershell
git status --short --branch
git fetch origin
git status --short --branch

git clone https://github.com/eantcal/nunn.wiki.git ..\nunn.wiki
git -C ..\nunn.wiki status --short --branch
```

The source repository uses `main`. The GitHub Wiki repository currently uses `master`. Do not assume that both branch names match.

## Publish the source repository

```powershell
git add README.md wiki scripts\check-wiki.ps1
git diff --cached
git commit -m "Improve source-linked wiki documentation"
git push origin main
```

Stage only the intended files. Existing unrelated untracked content is not part of a wiki publication.

## Synchronize the GitHub Wiki clone

Copy pages and assets into the clean wiki clone:

```powershell
Copy-Item -LiteralPath .\wiki\*.md -Destination ..\nunn.wiki -Force
Copy-Item -LiteralPath .\wiki\assets\* -Destination ..\nunn.wiki\assets -Force
```

Then inspect the result before committing:

```powershell
git -C ..\nunn.wiki status --short
git -C ..\nunn.wiki diff --check
git -C ..\nunn.wiki diff
```

If a source page or asset was intentionally deleted, remove the corresponding tracked file from the wiki clone explicitly. A plain copy does not delete stale remote files.

## Publish the GitHub Wiki

```powershell
git -C ..\nunn.wiki add --all
git -C ..\nunn.wiki commit -m "Improve source-linked documentation"
git -C ..\nunn.wiki push origin master
```

Finally, verify the rendered [wiki home](https://github.com/eantcal/nunn/wiki), sidebar, images, internal navigation, and several source links.

## Asset policy

`wiki/assets/` contains PNG diagrams adapted from book-generated figures. Keep their relative paths and descriptive names stable because both the source copy and public wiki refer to them.

Before adding a new diagram:

- confirm that it explains a relationship the prose or a short table cannot;
- crop unnecessary whitespace;
- use readable labels at GitHub's normal page width;
- add meaningful alt text;
- retain the source-generation material outside the wiki if it exists.

## Keep reading

Return to [Home](Home) to review the reader-facing navigation or [Implementation Map](Implementation-Map) to check source coverage.
