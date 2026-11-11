#!/usr/bin/env python3
"""Build the static Speech Bench site from a results checkout: the hub with its task pages, the highlights page and
one dashboard per machine.

The site is plain HTML for GitHub Pages: index.html is the hub, highlights.html the highlights page and
machines/<host>/ the dashboard of each machine with its data files. Links between the pages are relative, or
absolute under --base-url when the address the site is served from is known.

usage: build-site.py --results DIR --out DIR [--base-url URL]
"""
import argparse
import html
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
DASHBOARD_DIR = HERE / "dashboard"
DASHBOARDS_FILE = HERE / "dashboards.json"
DASHBOARD_FRAGMENT = "index.html"
DASHBOARD_SHELL = "static.html"
DASHBOARD_ASSETS = ("app.js", "app.css", "static-runtime.js", "static-theme.css")
DATASETS = ("runs", "history", "coverage", "machine", "commentary", "engines")
DATA_SUBDIR = "data"
ONE_WRITER_META = '<meta name="dash-one-writer">'
TITLE_TOKEN = "__TITLE__"
DASHBOARD_TO_HUB = "../../"
HUB_PAGE = "index.html"
HIGHLIGHTS_PAGE = "highlights.html"
MACHINES_DIR = "machines"
LANDING_KEY = "landing"
NO_JEKYLL = ".nojekyll"
BODY_TOKEN = "__BODY__"
PAGE_SKELETON = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<style>body{{margin:0}}img{{max-width:100%}}[hidden]{{display:none!important}}</style>
</head>
<body>
{BODY_TOKEN}
</body>
</html>
"""


def site_root(base_url):
    return base_url.rstrip("/") + "/" if base_url else ""


def hub_link(base_url):
    return site_root(base_url) or "./"


def machine_link(base_url, host):
    return f"{site_root(base_url)}{MACHINES_DIR}/{host}/"


def summary_hosts(results):
    return sorted(path.name for path in (results / "summaries").iterdir() if path.is_dir())


def site_links(hosts, base_url):
    return {LANDING_KEY: hub_link(base_url), **{host: machine_link(base_url, host) for host in hosts}}


def run_builder(script, args):
    try:
        return subprocess.run([sys.executable, str(HERE / script), *args], check=True, capture_output=True, text=True).stdout
    except subprocess.CalledProcessError as error:
        raise SystemExit(f"{script} failed: {error.stderr.strip()}") from error


def is_full_document(text):
    return text.lstrip().lower().startswith("<!doctype")


def as_document(text):
    return text if is_full_document(text) else PAGE_SKELETON.replace(BODY_TOKEN, text)


def write_document(path, text):
    path.write_text(as_document(text), encoding="utf-8")


def build_dashboard_data(results, links_file, data_dir):
    run_builder("build-dashboard-data.py", [str(results / "summaries"), "-o", str(data_dir), "--dashboards", str(links_file)])


def build_hub(data_dir, work, out):
    fragment = work / HUB_PAGE
    run_builder("build-landing.py", [str(data_dir / "hub.json"), "-o", str(fragment)])
    write_document(out / HUB_PAGE, fragment.read_text(encoding="utf-8"))


def build_highlights(data_dir, work, out):
    page = work / HIGHLIGHTS_PAGE
    run_builder("build-highlights.py", [str(data_dir), "-o", str(page)])
    write_document(out / HIGHLIGHTS_PAGE, page.read_text(encoding="utf-8"))


def legacy_hub_url():
    return json.loads(DASHBOARDS_FILE.read_text()).get(LANDING_KEY, "")


def dashboard_hub_link(base_url):
    return site_root(base_url) or DASHBOARD_TO_HUB


def dashboard_body(fragment, hub):
    body = fragment.replace(ONE_WRITER_META, "")
    legacy = legacy_hub_url()
    return body.replace(legacy, hub) if legacy else body


def dashboard_page(machine_label, hub):
    shell = (DASHBOARD_DIR / DASHBOARD_SHELL).read_text(encoding="utf-8")
    fragment = (DASHBOARD_DIR / DASHBOARD_FRAGMENT).read_text(encoding="utf-8")
    return shell.replace(TITLE_TOKEN, html.escape(f"{machine_label} · Speech Bench")).replace(BODY_TOKEN, dashboard_body(fragment, hub))


def copy_assets(target):
    for name in DASHBOARD_ASSETS:
        shutil.copyfile(DASHBOARD_DIR / name, target / name)


def copy_datasets(source, target):
    target.mkdir(parents=True, exist_ok=True)
    for name in DATASETS:
        if (source / f"{name}.json").is_file():
            shutil.copyfile(source / f"{name}.json", target / f"{name}.json")


def build_machine(machine, data_dir, out, base_url):
    target = out / MACHINES_DIR / machine["host"]
    target.mkdir(parents=True, exist_ok=True)
    copy_assets(target)
    copy_datasets(data_dir / machine["host"], target / DATA_SUBDIR)
    (target / DASHBOARD_FRAGMENT).write_text(dashboard_page(machine["machine"], dashboard_hub_link(base_url)), encoding="utf-8")


def build_machines(machines, data_dir, out, base_url):
    for machine in machines:
        build_machine(machine, data_dir, out, base_url)


def prepare_out(out):
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; build the site into a new or empty folder")
    out.mkdir(parents=True, exist_ok=True)


def hub_machines(data_dir):
    return json.loads((data_dir / "hub.json").read_text())["machines"]


def build(results, out, base_url):
    prepare_out(out)
    with tempfile.TemporaryDirectory() as tmp:
        work = pathlib.Path(tmp)
        links_file = work / "links.json"
        links_file.write_text(json.dumps(site_links(summary_hosts(results), base_url)))
        data_dir = work / "dashboard-data"
        build_dashboard_data(results, links_file, data_dir)
        build_hub(data_dir, work, out)
        build_highlights(data_dir, work, out)
        machines = hub_machines(data_dir)
        build_machines(machines, data_dir, out, base_url)
    (out / NO_JEKYLL).write_text("")
    return machines


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", required=True, help="checkout of the results repository (summaries/, report-rows.json)")
    parser.add_argument("--out", required=True, help="new or empty folder for the site")
    parser.add_argument("--base-url", default="", help="address the site is served from, for absolute links")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    machines = build(pathlib.Path(args.results), pathlib.Path(args.out), args.base_url)
    print(f"wrote the site to {args.out}: the hub, the highlights page and the dashboards of {len(machines)} machines")


if __name__ == "__main__":
    main()
