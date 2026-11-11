#!/usr/bin/env python3
"""Decide whether the results workflow publishes the site, from the machines that delivered this week's run.

A machine has delivered when summaries/<host>/<Sunday>.json or failures/<host>/<Sunday>.txt exists for the most
recent Sunday (UTC) on or before today. A push publishes only once every expected machine has delivered; the Monday
schedule and a manual run publish whatever has arrived. Prints publish, week and missing as key=value lines for
$GITHUB_OUTPUT and, with --summary, appends a Markdown table of the machines to that file.

usage: check-delivery.py --results DIR --event push|schedule|workflow_dispatch [--today YYYY-MM-DD]
                         [--hosts a,b] [--summary FILE]
"""
import argparse
import datetime
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
DASHBOARDS_FILE = HERE / "dashboards.json"
LANDING_KEY = "landing"
PUSH_EVENT = "push"
SUNDAY_WEEKDAY = 6
DAYS_PER_WEEK = 7
DELIVERED_SUMMARY = "summary"
DELIVERED_FAILURE = "failure notice"
MISSING = "missing"
STATE_LABELS = {DELIVERED_SUMMARY: "run summary", DELIVERED_FAILURE: "failure notice", MISSING: "nothing yet"}


def week_sunday(today):
    return today - datetime.timedelta(days=(today.weekday() - SUNDAY_WEEKDAY) % DAYS_PER_WEEK)


def split_list(text):
    return [item for item in (part.strip() for part in text.split(",")) if item]


def dashboard_hosts(path):
    return [key for key in json.loads(pathlib.Path(path).read_text()) if key != LANDING_KEY]


def expected_hosts(hosts_arg, dashboards_path):
    return split_list(hosts_arg) if hosts_arg else dashboard_hosts(dashboards_path)


def delivery_state(results, host, week):
    if (results / "summaries" / host / f"{week}.json").is_file():
        return DELIVERED_SUMMARY
    if (results / "failures" / host / f"{week}.txt").is_file():
        return DELIVERED_FAILURE
    return MISSING


def delivery_states(results, hosts, week):
    return {host: delivery_state(results, host, week) for host in hosts}


def missing_hosts(states):
    return [host for host, state in states.items() if state == MISSING]


def should_publish(event, states):
    return event != PUSH_EVENT or not missing_hosts(states)


def output_lines(publish, week, states):
    return [f"publish={str(publish).lower()}", f"week={week}", f"missing={','.join(missing_hosts(states))}"]


def decision_text(event, publish, states):
    if not publish:
        return f"Not published: waiting for {', '.join(missing_hosts(states))}."
    if event == PUSH_EVENT:
        return "Published: every machine has delivered."
    return f"Published by the {event} run with what has arrived."


def summary_markdown(week, states, event, publish):
    rows = [f"| {host} | {STATE_LABELS[state]} |" for host, state in states.items()]
    return "\n".join([f"## Speech bench, week of {week}", "", "| machine | delivered |", "| --- | --- |", *rows, "",
                      decision_text(event, publish, states), ""])


def append_summary(path, text):
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(text + "\n")


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", required=True)
    parser.add_argument("--event", required=True)
    parser.add_argument("--today", default="")
    parser.add_argument("--hosts", default="")
    parser.add_argument("--dashboards", default=str(DASHBOARDS_FILE))
    parser.add_argument("--summary", default="")
    return parser.parse_args(argv)


def today_of(text):
    return datetime.date.fromisoformat(text) if text else datetime.datetime.now(datetime.timezone.utc).date()


def main(argv=None):
    args = parse_args(argv)
    week = week_sunday(today_of(args.today)).isoformat()
    states = delivery_states(pathlib.Path(args.results), expected_hosts(args.hosts, args.dashboards), week)
    publish = should_publish(args.event, states)
    print("\n".join(output_lines(publish, week, states)))
    if args.summary:
        append_summary(args.summary, summary_markdown(week, states, args.event, publish))


if __name__ == "__main__":
    main()
