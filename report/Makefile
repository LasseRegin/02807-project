#XELATEX='xelatex -file-line-error -interaction=nonstopmode'
.PHONY: main.pdf all clean

all: main.pdf

main.pdf: main.tex
	latexmk -pdf -pdflatex="xelatex -file-line-error -interaction=nonstopmode" -use-make main.tex

auto: main.tex
	latexmk -pdf -pdflatex="xelatex -file-line-error -interaction=nonstopmode" -use-make -pvc main.tex

clean:
	latexmk -CA
