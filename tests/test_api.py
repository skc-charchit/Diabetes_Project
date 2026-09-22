import unittest

from fastapi.testclient import TestClient

from main import app


class ApiContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.payload = {
            "Pregnancies": 6,
            "Glucose": 148,
            "BloodPressure": 72,
            "SkinThickness": 35,
            "Insulin": 125,
            "BMI": 33.6,
            "DiabetesPedigreeFunction": 0.627,
            "Age": 50,
        }

    def test_health_reports_loaded_model(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")
        self.assertTrue(response.json()["artifact_sha256"])
        self.assertEqual(response.json()["pipeline_version"], "1.0.0")

    def test_prediction_contract(self):
        response = self.client.post("/diabetes_prediction", json=self.payload)

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn(body["prediction"], (0, 1))
        self.assertGreaterEqual(body["probability"], 0)
        self.assertLessEqual(body["probability"], 1)

    def test_invalid_measurement_is_rejected(self):
        payload = {**self.payload, "Glucose": 400}

        response = self.client.post("/diabetes_prediction", json=payload)

        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
