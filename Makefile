# Makefile

.PHONY: all build-dashboard build-consumer build-producer

all: build-dashboard build-consumer build-producer

build-producer:
	docker build -t open_fair-producer -f producer/Dockerfile .

build-consumer:
	docker build -t open_fair-consumer -f consumer/Dockerfile .
