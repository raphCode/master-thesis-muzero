# latexmk: coordinate compiling tool runs
# texfot: closes stdin to remove these dang input prompts, also filters only relevant
# output

clean:
	latexmk -C *.tex
	rm -f *.bbl *.tdo

pvc:
	texfot latexmk -pvc

check:
	checkcites main
