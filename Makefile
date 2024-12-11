# Run / Stop docker container.
deploy_jupyter:
	docker build -f Dockerfile -t jupyter_environment .
	docker compose -f docker-compose.yml up

jupyter_up:
	docker compose -f docker-compose.yml up

jupyter_down:
	docker compose -f docker-compose.yml down
