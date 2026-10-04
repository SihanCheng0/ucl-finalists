.PHONY: help setup web dev test pipeline report site

help:  ## List the commands
	@grep -E '^[a-z]+:.*## ' Makefile | awk 'BEGIN {FS = ":.*## "} {printf "  make %-9s %s\n", $$1, $$2}'

setup:  ## Install the Python and web dependencies, then build the dashboard
	uv sync
	npm --prefix web ci
	npm --prefix web run build

web:  ## Serve UCL Lab at http://127.0.0.1:8787 and open it
	uv run ucl web

dev:  ## Frontend hot reload on :5173 (run `uv run ucl web --no-open` alongside)
	npm --prefix web run dev

test:  ## Run the Python and frontend tests
	uv run pytest -q
	npm --prefix web test

pipeline:  ## Run every stage: fetch, build, model, analyze, report
	uv run ucl all

report:  ## Rebuild out/report.html from the saved outputs
	uv run ucl report

site:  ## Write the website into .vercel/output: its frontend build and every page's data
	npm --prefix web run build:site
	uv run ucl publish
