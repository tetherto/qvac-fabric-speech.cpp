#!/usr/bin/env python3
import importlib.util
import json
import pathlib
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


site = load_module("build_site", "build-site.py")
fixtures = load_module("dashboard_fixtures", "test_build_dashboard_data.py")
HOSTS = ("qvac-dev-linux-x64", "spark-d9ca")
LEGACY_HUB = "claude.ai/artifact"


def write_summary(results, host, name):
    doc = fixtures.summary(name, f"{name}T17:00:00Z", [fixtures.result("cpu", "parakeet-tdt", 12.0), fixtures.result("cuda", "audio8-lm-f16-dec-f16", 2.0)])
    doc["run"]["host"] = host
    path = results / "summaries" / host / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))


class BuildSiteTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = pathlib.Path(cls.tmp.name)
        cls.results = root / "results"
        for host in HOSTS:
            write_summary(cls.results, host, "2026-11-08")
        cls.relative = root / "site"
        cls.machines = site.build(cls.results, cls.relative, "")
        cls.absolute = root / "site-abs"
        site.build(cls.results, cls.absolute, "https://speech.pages.example/bench")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def page(self, root, name):
        return (root / name).read_text(encoding="utf-8")

    def test_the_hub_is_a_full_page_with_its_task_views_and_site_links(self):
        hub = self.page(self.relative, "index.html")
        self.assertTrue(hub.startswith("<!doctype html>"))
        self.assertIn("[hidden]{display:none!important}", hub)
        self.assertIn('data-view="rtx-5090-asr"', hub)
        self.assertIn('href="machines/qvac-dev-linux-x64/"', hub)
        self.assertIn('href="machines/spark-d9ca/"', hub)
        self.assertNotIn(LEGACY_HUB, hub)

    def test_the_highlights_page_links_back_to_the_site_hub(self):
        highlights = self.page(self.relative, "highlights.html")
        self.assertTrue(highlights.startswith("<!doctype html>"))
        self.assertEqual(highlights.count("<!doctype html>"), 1)
        self.assertIn('href="./"', highlights)
        self.assertNotIn(LEGACY_HUB, highlights)

    def test_a_base_url_makes_every_site_link_absolute(self):
        hub = self.page(self.absolute, "index.html")
        self.assertIn('href="https://speech.pages.example/bench/machines/spark-d9ca/"', hub)
        self.assertIn("https://speech.pages.example/bench/#dgx-spark-tts", hub)
        self.assertIn('href="https://speech.pages.example/bench/"', self.page(self.absolute, "highlights.html"))

    def test_the_site_lists_every_machine_with_summaries_and_turns_jekyll_off(self):
        self.assertEqual(sorted(m["host"] for m in self.machines), sorted(HOSTS))
        self.assertTrue((self.relative / ".nojekyll").is_file())

    def test_each_machine_gets_a_standalone_dashboard_with_its_data_files(self):
        folder = self.relative / "machines" / "qvac-dev-linux-x64"
        page = self.page(folder, "index.html")
        self.assertTrue(page.startswith("<!doctype html>"))
        self.assertIn("<title>RTX 5090 · Speech Bench</title>", page)
        self.assertIn('integrity="sha384-', page)
        self.assertLess(page.index('src="static-runtime.js"'), page.index('src="app.js"'))
        self.assertIn('data-draw="trend"', page)
        self.assertIn('href="../../"', page)
        self.assertNotIn(LEGACY_HUB, page)
        self.assertNotIn("dash-one-writer", page)
        for name in site.DASHBOARD_ASSETS:
            self.assertTrue((folder / name).is_file(), name)
        for name in ("runs", "history", "coverage", "machine", "commentary", "engines"):
            self.assertTrue((folder / "data" / f"{name}.json").is_file(), name)
        machine = json.loads((folder / "data" / "machine.json").read_text())
        self.assertEqual(machine[0]["machine"], "RTX 5090")

    def test_a_base_url_points_the_dashboards_at_the_absolute_hub(self):
        page = self.page(self.absolute / "machines" / "spark-d9ca", "index.html")
        self.assertIn('href="https://speech.pages.example/bench/"', page)
        self.assertIn("<title>DGX Spark · Speech Bench</title>", page)

    def test_a_folder_that_is_not_empty_is_refused_and_left_alone(self):
        with self.assertRaises(SystemExit):
            site.build(self.results, self.relative, "")
        self.assertTrue((self.relative / "index.html").is_file())


if __name__ == "__main__":
    unittest.main()
