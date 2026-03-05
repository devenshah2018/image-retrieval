
## Useful Commands
`docker run --name retrieval-webapp -p 9999:9999 -p 8501:8501 -v "$(pwd)":/image-retrieval -it image-retrieval-env`

`docker start -ai retrieval-webapp`

`cd /Users/devenshah/repos/image-retrieval && .venv/bin/pip install -r requirements.txt`