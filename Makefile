.PHONY: checktype
checktype:
	mypy --ignore-missing-imports src/

.PHONY: install
install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

.PHONY: uninstall
uninstall:
	pip install --upgrade pip &&\
		pip uninstall -r requirements.txt -y

.PHONY: format
format:
	black -l 79 src/*.py

.PHONY: lint
lint:
	pylint --disable=R,C src/*.py

.PHONY: qtconsole
qtconsole:
	jupyter qtconsole --kernel=lstm_xgboost &

.PHONY: jupyter
jupyter:
	jupyter notebook --kernel=lstm_xgboost &

.PHONY: test
test:
	pytest -n auto -vv -s tests

.PHONY: clean
clean:
	find . -type f -name "*.pyc" | xargs rm -rf
	find . -type d -name __pycache__ | xargs rm -rf

.PHONY: checklist
checklist: lint typehint

.PHONY: all
all: install lint typehint format
