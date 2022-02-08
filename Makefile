# latexmk: coordinate compiling tool runs

clean:
	latexmk -C *.tex
	rm -f *.bbl *.tdo

pvc:
	latexmk -pvc <&-

check:
	checkcites main
