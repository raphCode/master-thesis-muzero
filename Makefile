# latexmk: coordinate compiling tool runs

clean:
	latexmk -C *.tex
	rm -f *.bbl *.tdo

continous-preview:
	latexmk -pvc <&-

pdf:
	latexmk <&-

check:
	checkcites main
