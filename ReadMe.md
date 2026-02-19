
## Useful Commands
`docker run --name retrieval-webapp -p 9999:9999 -p 8501:8501 -v "$(pwd)":/image-retrieval -it image-retrieval-env`

`docker start -ai retrieval-webapp`