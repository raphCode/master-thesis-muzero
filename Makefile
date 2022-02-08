# latexmk: coordinate compiling tool runs

clean:
	latexmk -C *.tex
	rm -f *.bbl *.tdo

continous-preview:
	latexmk -pvc <&-

check:
	checkcites main
