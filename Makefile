

install:
	pip install -r requirements.txt

install jupyter-kernel:
	source venv/bin/activate && pip install ipykernel && python -m ipykernel install --user --name kernel_desafio

run:
	jupyter notebook solution.ipynb

