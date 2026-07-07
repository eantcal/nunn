# Publishing the Wiki

The files in this directory are a source copy of the nuNN wiki.

GitHub stores project wikis in a separate repository:

```text
https://github.com/eantcal/nunn.wiki.git
```

If that repository exists, publish the wiki with:

```sh
git clone https://github.com/eantcal/nunn.wiki.git nunn.wiki
cp -r wiki/* nunn.wiki/
cd nunn.wiki
git add .
git commit -m "Update nuNN wiki"
git push
```

If the wiki repository does not exist yet, enable or initialize the GitHub Wiki first from the repository page, then repeat the commands above.

The `assets/` directory contains PNG diagrams adapted from the book-generated TikZ figures. Keep the descriptive filenames stable so existing wiki pages do not break.

