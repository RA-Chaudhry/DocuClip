import unittest
from fastapi.testclient import TestClient

from app.main import app


class SmokeTests(unittest.TestCase):
    def test_health_endpoint(self):
        client = TestClient(app)
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "healthy")


if __name__ == "__main__":
    unittest.main()
