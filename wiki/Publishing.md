# Publishing the Wiki

The files in this directory are a source copy of the nuNN wiki.

GitHub stores project wikis in a separate repository:

```text
https://github.com/eantcal/nunn.wiki.git
```

Publish the wiki with:

```sh
git clone https://github.com/eantcal/nunn.wiki.git nunn.wiki
cp -r wiki/* nunn.wiki/
cd nunn.wiki
git add .
git commit -m "Update nuNN wiki"
git push
```

The wiki repository currently exists as `eantcal/nunn.wiki.git`. No GitHub Action is required to make the wiki active: pushing to that repository updates the GitHub Wiki directly.

The `assets/` directory contains PNG diagrams adapted from the book-generated TikZ figures. Keep the descriptive filenames stable so existing wiki pages do not break.
