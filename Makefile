

install:
	pip install -r requirements.txt

install_jupyter_kernel:
	. venv/bin/activate && pip install ipykernel && python -m ipykernel install --user --name kernel_desafio

run:
	jupyter notebook solution.ipynb


