# Makefile

.PHONY: all build-dashboard build-consumer build-producer all_scache

all: build-dashboard build-consumer build-producer build-wandber

all_scache: build-dashboard-scache build-producer-scache build-consumer-scache build-wandber-scache

build-producer:
	docker build -t open_fair-producer -f producer/Dockerfile .

build-consumer:
	docker build -t open_fair-consumer -f consumer/Dockerfile .

build-dashboard:
	docker build -t open_fair-dashboard -f dashboard/Dockerfile .

build-wandber:
	docker build -t open_fair-wandber -f wandber/Dockerfile .

build-producer-scache:
	docker build --build-arg CACHE_BUST=$(shell date +%s) -t open_fair-producer -f producer/Dockerfile .

build-consumer-scache:
	docker build --build-arg CACHE_BUST=$(shell date +%s) -t open_fair-consumer -f consumer/Dockerfile .

build-dashboard-scache:
	docker build --build-arg CACHE_BUST=$(shell date +%s) -t open_fair-dashboard -f dashboard/Dockerfile .

build-wandber-scache:
	docker build --build-arg CACHE_BUST=$(shell date +%s) -t open_fair-wandber -f wandber/Dockerfile .