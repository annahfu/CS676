See server.py for API, templates/chat_adapter.py for chatbot adapter, tests/ for unit & integration tests, and Dockerfile for deployment.


## Frontend widget
A minimal, dependency-free widget is provided under `/web`:
- `credibility_widget.js`, `credibility_widget.css`, and `example.html`.
To use in your chat UI, load the JS/CSS and call:
```js
renderCredibilityList(document.getElementById('your-container'), resultsArray);
```

## CI (GitHub Actions)
A basic CI pipeline is included in `.github/workflows/ci.yml`:
- Runs tests on push/PR.
- Optionally builds + pushes a Docker image when `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` secrets are set.
