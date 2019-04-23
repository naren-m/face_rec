.PHONY: help build clean vm-up vm-destroy vm-reset

.DEFAULT: help
help:
	@echo "make test       run tests"
	@echo "make format     run yapf formatter"

test: clean test-all coverage-html

test-all:
	pytest --cov=ztp2-dev --duration=3 test

coverage: test coverage-html coverage-xml coverage-report

coverage-report:
	coverage report

coverage-html:
	coverage html

coverage-xml:
	coverage xml

clean-coverage:
	rm -rf htmlcov

clean-logs:
	rm -f /var/log/ztp.log

clean-pyc: ## remove Python file artifacts
		find . -name '*.pyc' -exec rm -f {} +
		find . -name '*.pyo' -exec rm -f {} +
		find . -name '*~' -exec rm -f {} +
		find . -name '__pycache__' -exec rm -rf {} +

clean-build:
	rm -rf tools/bin/zpyinstaller/__ztemp__

clean: clean-build clean-pyc clean-coverage
		find . -name '*.swp' -exec rm -rf {} +
		find . -name '*.bak' -exec rm -rf {} +

format: clean
	yapf -r -i .

