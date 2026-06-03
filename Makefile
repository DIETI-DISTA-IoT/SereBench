# Makefile

.PHONY: all build-dashboard build-consumer build-producer build-wandber \
        all-scache all-scache-nolib \
        build-dashboard-scache build-consumer-scache build-producer-scache build-wandber-scache \
        build-dashboard-scache-nolib build-consumer-scache-nolib build-producer-scache-nolib build-wandber-scache-nolib

all: build-dashboard build-consumer build-producer build-wandber

# Full rebuild with fresh code + fresh pip install
all-scache: build-dashboard-scache build-producer-scache build-consumer-scache build-wandber-scache

# Fresh code pull only — pip install layers stay cached (faster when requirements.txt is unchanged)
all-scache-nolib: build-dashboard-scache-nolib build-producer-scache-nolib build-consumer-scache-nolib build-wandber-scache-nolib

build-producer:
	docker build -t open_fair-producer -f producer/Dockerfile .

build-consumer:
	docker build -t open_fair-consumer -f consumer/Dockerfile .

build-dashboard:
	docker build -t open_fair-dashboard -f dashboard/Dockerfile .

build-wandber:
	docker build -t open_fair-wandber -f wandber/Dockerfile .

# scache: busts pip install AND code clone (use when requirements.txt may have changed)
build-producer-scache:
	docker build --build-arg CACHE_BUST=$(shell date +%s) -t open_fair-producer -f producer/Dockerfile .

build-consumer-scache:
	docker build --build-arg CACHE_BUST=$(shell date +%s) -t open_fair-consumer -f consumer/Dockerfile .

build-dashboard-scache:
	docker build --build-arg CACHE_BUST=$(shell date +%s) -t open_fair-dashboard -f dashboard/Dockerfile .

build-wandber-scache:
	docker build --build-arg CACHE_BUST=$(shell date +%s) -t open_fair-wandber -f wandber/Dockerfile .

# scache-nolib: busts only the git clones, reuses cached pip layers (use when only code changed)
build-producer-scache-nolib:
	docker build --build-arg CODE_BUST=$(shell date +%s) -t open_fair-producer -f producer/Dockerfile .

build-consumer-scache-nolib:
	docker build --build-arg CODE_BUST=$(shell date +%s) -t open_fair-consumer -f consumer/Dockerfile .

build-dashboard-scache-nolib:
	docker build --build-arg CODE_BUST=$(shell date +%s) -t open_fair-dashboard -f dashboard/Dockerfile .

build-wandber-scache-nolib:
	docker build --build-arg CODE_BUST=$(shell date +%s) -t open_fair-wandber -f wandber/Dockerfile .
