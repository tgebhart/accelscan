# accelscan-site

The public arXiv-results website: 7 headline charts (Plotly.js) plus a
paper drill-down browser (DuckDB-WASM querying the published parquet directly
in the visitor's browser). No backend — everything here builds to static
files.

Data is **not** built by this project; it comes from
`python -m accelscan.scripts.export_site_data --out ../site_data` (run from
the repo root, with a `.env` sourced for MSI S3 credentials — see
`docs/site_deploy.md`). During local development, symlink or copy that output
under `site/public/data/` so Vite serves it at `/data/...`:

```
python -m accelscan.scripts.export_site_data --out site_data
ln -s ../../site_data site/public/data
```

## Setup

This machine did not have Node/npm installed while these files were authored
(MSI module system). Load a Node module or run this elsewhere first, e.g.:

```
module load nodejs   # or: nvm install --lts
```

Then:

```
cd site
npm install
npm run dev       # local dev server, http://localhost:5173
npm run build     # writes static output to site/dist/
```

`dist/` is what gets synced to S3 — see `../docs/site_deploy.md` for the full
deploy (and monthly redeploy) commands.
