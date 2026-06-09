# AWS Deployment Strategy

This document explains how the Aero RUL Predictor API can be deployed to AWS.

The project exposes a trained machine learning model through a FastAPI service. The API is already containerized using Docker, so it can be deployed to AWS using multiple approaches.

Recommended deployment options:

1. Amazon EC2
2. Amazon Elastic Container Registry plus Amazon ECS
3. AWS Elastic Beanstalk
4. Amazon SageMaker endpoint

For the first production-style portfolio deployment, EC2 is the simplest and most practical option.

## Recommended First Deployment: AWS EC2

Amazon EC2 is a good first deployment target because it allows full control over the server, Docker runtime, ports, security groups, and deployment process.

Deployment flow:

```text
Local project
→ Push code to GitHub
→ Launch EC2 instance
→ Install Docker on EC2
→ Clone GitHub repository
→ Copy trained model artifact
→ Build Docker image on EC2
→ Run FastAPI container
→ Access API through EC2 public IP
```

This approach demonstrates practical ML deployment knowledge without adding too much cloud complexity too early.

## Actual EC2 Deployment Performed

The API was deployed manually on an Ubuntu EC2 instance using Docker.

### EC2 Configuration Used

```text
Region: ap-south-1
Operating system: Ubuntu Server
Instance type: t3.small
Storage: 16 GiB gp3
Public IP: Enabled
Security group:
  - SSH port 22 allowed from My IP
  - API port 8000 allowed from My IP
```

### Server Setup

After connecting to the EC2 instance through SSH, the server package list was updated.

```bash
sudo apt update
```

Docker was installed on the Ubuntu EC2 instance using Docker's official repository-based installation flow.

Docker installation was verified using:

```bash
docker --version
sudo docker run hello-world
```

### Project Deployment Flow

The GitHub repository was cloned into the EC2 instance.

```bash
git clone https://github.com/kallurivenkatesh4416-commits/aero-rul-predictor.git
cd aero-rul-predictor
```

Because the trained model artifact is intentionally not committed to GitHub, the model file was copied separately from the local machine to the EC2 instance.

```text
models/xgboost_rul_model.joblib
```

The Docker image was then built on the EC2 instance.

```bash
sudo docker build -t aero-rul-api .
```

The FastAPI application container was started on port 8000.

```bash
sudo docker run -d --name aero-rul-api-container -p 8000:8000 aero-rul-api
```

The running container was verified using:

```bash
sudo docker ps
```

### API Verification

The API was tested from inside the EC2 instance.

```bash
curl http://localhost:8000/
```

Expected response:

```json
{
  "status": "ok",
  "message": "Aircraft engine RUL prediction API is running"
}
```

The API documentation was also verified from the browser.

```text
http://<EC2_PUBLIC_IP>:8000/docs
```

The `/predict` endpoint was tested successfully and returned a valid Remaining Useful Life prediction with a risk category.

Example response:

```json
{
  "predicted_rul": 120.07,
  "risk_category": "Healthy"
}
```

## Model Artifact Handling

The trained model file is not committed to GitHub because model artifacts can become large and should generally be handled separately from source code.

For this deployment, the trained model was copied manually to the EC2 instance.

```text
Local machine
→ models/xgboost_rul_model.joblib
→ EC2 instance
→ /home/ubuntu/aero-rul-predictor/models/xgboost_rul_model.joblib
```

In a more advanced production setup, the model artifact can be stored in Amazon S3, GitHub Releases, Amazon ECR image layers or a dedicated model registry.

## Future CI/CD Improvement

The current deployment was performed manually to understand full infrastructure flow.

A future improvement is to add GitHub Actions for CI/CD.

Possible CI/CD stages:

```text
Push code to GitHub
→ Run linting and tests
→ Build Docker image
→ Push image to container registry
→ Deploy updated container to EC2 or ECS
```

This would make the deployment more automated and production-ready.

## Notes

The EC2 public IP may change if the instance is stopped and started again. For a permanent production-style endpoint, an Elastic IP or domain name should be configured.

This deployment demonstrates a complete manual ML deployment workflow using GitHub, Docker, AWS EC2, FastAPI, and a trained XGBoost model artifact.
