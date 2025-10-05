from locust import HttpUser, task, between

class CredUser(HttpUser):
    wait_time = between(0.1, 0.5)
    @task(5)
    def score(self):
        self.client.post('/score', json={'id':'x','text':'Peer reviewed methodology with dataset.','url':'https://example.edu'})
    @task(1)
    def score_batch(self):
        items = [{'id':str(i),'text':'Shocking miracle cure!!!','url':'https://blog.example.com'} for i in range(8)]
        self.client.post('/score_batch', json={'items':items})
