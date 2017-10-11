

all: fig4, fig5, fig6

fig4: ./fig/comparison_vary_closeness_linsep.dat ./fig/comparison_vary_closeness_extreme.dat
	cd ./tex; pdflatex fig4.tex

fig5: ./fig/comparison_varyn09.dat
	cd ./tex; pdflatex fig5.tex

fig6: ./dat/generalization_n10.dat
	cd ./tex; pdflatex fig5.tex




comparison_vary_closeness_linsep.dat:
	python -c 'from comparison_parametric import *; exp_fig4a()'

comparison_vary_closeness_extreme.dat:
	python -c 'from comparison_parametric import *; exp_fig4b()'

comparison_varyn09.dat:
	python -c 'from comparison_parametric import *; exp_fig5a()'

generalization_n10.dat:
	python -c 'from comparison_parametric import *; exp_fig6()'
	
clean:
	find ./tex/ -type f ! -name '*.tex' -delete
	cd ./dat; rm *.dat

