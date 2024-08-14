.PHONY: all
all:
	docker build . -t transform_artifact

run:
	docker run -it -w /home transform_artifact zsh


.PHONY: all