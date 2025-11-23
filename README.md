# Finlens AI - Transaction Categorization Backend

A powerful FastAPI-based backend system for intelligent transaction categorization using machine learning. This system automatically categorizes financial transactions using a hybrid neural network model that combines text embeddings with tabular features.

## 🚀 Features

- **AI-Powered Transaction Categorization**: Automatically categorizes transactions using a hybrid ML model
- **CSV Upload & Batch Processing**: Upload CSV files with transactions for bulk categorization
- **User Feedback System**: Users can provide feedback on categorizations to improve future predictions
- **Analytics Dashboard**: Get detailed analytics including inflows, outflows, and category breakdowns
- **Merchant Knowledge Base**: Semantic search using vector database (Qdrant) for merchant matching
- **JWT Authentication**: Secure user and admin authentication
- **RESTful API**: Well-documented API with OpenAPI/Swagger documentation

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [API Documentation](#api-documentation)
- [Project Structure](#project-structure)
- [Model Training](#model-training)
- [Usage Examples](#usage-examples)
- [Docker Deployment](#docker-deployment)
- [Technologies Used](#technologies-used)

## 🔧 Prerequisites

- Python 3.9 or higher
- MongoDB (local or cloud instance)
- Qdrant Vector Database (cloud or local instance)
- pip (Python package manager)

## 📦 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd finlens-ai-backend
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Install System Dependencies (if needed)

For PyTorch and sentence-transformers, you may need additional system libraries:

```bash
# On Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y gcc ffmpeg

# On macOS
brew install gcc ffmpeg
```

## ⚙️ Configuration

### 1. Create `.env` File

Create a `.env` file in the root directory with the following variables:

```env
# MongoDB Configuration
MONGO_URI=mongodb://localhost:27017
MONGO_DB=finlens_db

# JWT Configuration
JWT_SECRET_KEY=your-secret-key-here-change-in-production
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24

# Qdrant Vector Database Configuration
QDRANT_HOST=https://your-qdrant-instance.qdrant.io
QDRANT_API_KEY=your-qdrant-api-key
```

### 2. MongoDB Setup

Make sure MongoDB is running:

```bash
# Using Docker
docker run -d -p 27017:27017 --name mongodb mongo:latest

# Or using local MongoDB installation
mongod
```

### 3. Qdrant Setup

#### Option A: Cloud (Recommended for Hackathon)

1. Sign up at [Qdrant Cloud](https://cloud.qdrant.io/)
2. Create a cluster
3. Get your cluster URL and API key
4. Add them to `.env`

#### Option B: Local

```bash
# Using Docker
docker run -p 6333:6333 qdrant/qdrant
```

Then set in `.env`:
```env
QDRANT_HOST=http://localhost:6333
QDRANT_API_KEY=
```

## 🏃 Running the Application

### Development Mode

```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Run the application
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **OpenAPI Schema**: http://localhost:8000/openapi.json

### Production Mode

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## 📚 API Documentation

### Base URL
```
http://localhost:8000
```

### Authentication

Most endpoints require JWT authentication. Include the token in the Authorization header:

```
Authorization: Bearer <your-jwt-token>
```

### Public Endpoints

#### Health Check
```http
GET /health
```

#### User Signup
```http
POST /auth/signup
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "securepassword123",
  "name": "John Doe"
}
```

#### User Login
```http
POST /auth/login
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "securepassword123"
}
```

### Transaction Endpoints (Requires Authentication)

#### Upload Transactions CSV
```http
POST /transactions/upload-transactions
Authorization: Bearer <token>
Content-Type: multipart/form-data

file: <csv-file>
```

**CSV Format:**
```csv
description,amount,date,transaction_type
AMZN PYMT BLR,1937,2025-11-10,DEBIT
STARBUCKS COFFEE,250,2025-11-11,DEBIT
```

#### Get Analytics Summary
```http
GET /transactions/analytics?year=2024&month=11
Authorization: Bearer <token>
```

#### Get Transactions by Category
```http
GET /transactions/category-transactions?year=2024&month=11&category=Shopping&transaction_type=DEBIT
Authorization: Bearer <token>
```

#### Submit Category Feedback
```http
POST /transactions/{transaction_id}/feedback
Authorization: Bearer <token>
Content-Type: application/json

{
  "category": "Shopping"
}
```

#### Get Available Categories
```http
GET /transactions/categories
Authorization: Bearer <token>
```

### Admin Endpoints (Requires Admin Authentication)

#### Admin Signup
```http
POST /admin/auth/signup
Content-Type: application/json

{
  "email": "admin@example.com",
  "password": "adminpassword123",
  "name": "Admin User"
}
```

#### Admin Login
```http
POST /admin/auth/login
Content-Type: application/json

{
  "email": "admin@example.com",
  "password": "adminpassword123"
}
```

#### Train Model
```http
POST /admin/train-model?version=latest
Authorization: Bearer <admin-token>
Content-Type: multipart/form-data

file: <training-csv-file>
```

**Training CSV Format:**
```csv
description,amount,date,transaction_type,label
AMAZON PAYMENTS,899,2025-11-07,DEBIT,Shopping
STARBUCKS COFFEE,230,2025-11-07,DEBIT,Food & Beverage
```

#### Upload Merchant Mappings
```http
POST /admin/upload-merchant-mappings
Authorization: Bearer <admin-token>
Content-Type: multipart/form-data

file: <merchant-mappings-csv>
```

**Merchant Mappings CSV Format:**
```csv
Category,Merchants
Shopping,"Amazon, Amazon.in, AMZN"
Food & Beverage,"Starbucks, Cafe Coffee Day"
```

#### Upload Merchant Corrections
```http
POST /admin/upload-merchant-corrections
Authorization: Bearer <admin-token>
Content-Type: multipart/form-data

file: <corrections-csv>
```

**Corrections CSV Format:**
```csv
raw_description,canonical_merchant,canonical_category
AMZN PYMT BLR,Amazon,Shopping
```

## 📁 Project Structure

```
finlens-ai-backend/
├── app/
│   ├── api/
│   │   ├── routes.py                 # Main router configuration
│   │   └── v1/
│   │       ├── auth_controller.py    # User authentication endpoints
│   │       ├── transactions_controller.py  # Transaction endpoints
│   │       ├── admin_auth_controller.py    # Admin authentication
│   │       └── admin_controller.py   # Admin management endpoints
│   ├── core/
│   │   ├── config.py                 # Application configuration
│   │   ├── db.py                     # Database connection
│   │   ├── dependencies.py           # Dependency injection
│   │   ├── exceptions.py             # Custom exceptions
│   │   ├── handlers.py               # Exception handlers
│   │   ├── jwt_handler.py            # JWT utilities
│   │   ├── middleware.py             # Authentication middleware
│   │   └── admin_middleware.py       # Admin auth middleware
│   ├── ml/
│   │   └── transaction_categorization_engine.py  # ML model engine
│   ├── models/                       # Pydantic models
│   ├── repositories/                 # Data access layer
│   ├── schemas/                      # Request/Response schemas
│   ├── services/                     # Business logic layer
│   └── main.py                       # FastAPI application entry point
├── Train/                            # Training data files
├── requirements.txt                  # Python dependencies
├── Dockerfile                        # Docker configuration
└── README.md                        # This file
```

## 🤖 Model Training

### Quick Start Training

1. **Prepare Training Data**

   Create a CSV file with the following columns:
   - `description`: Transaction description
   - `amount`: Transaction amount
   - `date` or `timestamp`: Transaction date
   - `transaction_type`: DEBIT or CREDIT (optional)
   - `label` or `category`: Target category for training

2. **Upload Training Data**

   Use the admin endpoint to train the model:
   ```bash
   curl -X POST "http://localhost:8000/admin/train-model?version=latest" \
     -H "Authorization: Bearer <admin-token>" \
     -F "file=@Train/training_transactions.csv"
   ```

3. **Model Architecture**

   The model uses:
   - **Text Features**: SentenceTransformer embeddings (all-MiniLM-L6-v2)
   - **Tabular Features**: Amount, time features, transaction type, merchant confidence
   - **Hybrid Neural Network**: Two-pathway network combining text and tabular features
   - **Merchant Knowledge Base**: Vector database for semantic merchant matching

### Training Data Requirements

- Minimum 2 different categories
- At least 1 sample per category
- Recommended: 50+ samples for global model, 20+ for user-specific model

## 💡 Usage Examples

### Example 1: Upload and Categorize Transactions

```python
import requests

# Login
response = requests.post("http://localhost:8000/auth/login", json={
    "email": "user@example.com",
    "password": "password123"
})
token = response.json()["data"]["access_token"]

# Upload CSV
headers = {"Authorization": f"Bearer {token}"}
with open("transactions.csv", "rb") as f:
    files = {"file": f}
    response = requests.post(
        "http://localhost:8000/transactions/upload-transactions",
        headers=headers,
        files=files
    )
print(response.json())
```

### Example 2: Get Analytics

```python
import requests

headers = {"Authorization": f"Bearer {token}"}
response = requests.get(
    "http://localhost:8000/transactions/analytics?year=2024&month=11",
    headers=headers
)
analytics = response.json()["data"]
print(f"Total Inflows: {analytics['total_inflows']}")
print(f"Total Outflows: {analytics['total_outflows']}")
```

### Example 3: Submit Feedback

```python
import requests

headers = {"Authorization": f"Bearer {token}"}
response = requests.post(
    f"http://localhost:8000/transactions/{transaction_id}/feedback",
    headers=headers,
    json={"category": "Shopping"}
)
print(response.json())
```

## 🐳 Docker Deployment

### Build Docker Image

```bash
docker build -t finlens-ai-backend .
```

### Run Docker Container

```bash
docker run -d \
  -p 8000:8080 \
  --name finlens-backend \
  --env-file .env \
  finlens-ai-backend
```

### Docker Compose (Optional)

Create a `docker-compose.yml`:

```yaml
version: '3.8'

services:
  backend:
    build: .
    ports:
      - "8000:8080"
    env_file:
      - .env
    depends_on:
      - mongodb
      - qdrant

  mongodb:
    image: mongo:latest
    ports:
      - "27017:27017"
    volumes:
      - mongodb_data:/data/db

  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

volumes:
  mongodb_data:
  qdrant_data:
```

Run with:
```bash
docker-compose up -d
```

## 🛠️ Technologies Used

### Backend Framework
- **FastAPI**: Modern, fast web framework for building APIs
- **Uvicorn**: ASGI server for running FastAPI

### Database
- **MongoDB**: NoSQL database for transaction storage
- **GridFS**: For storing ML models
- **Qdrant**: Vector database for semantic merchant search

### Machine Learning
- **PyTorch**: Deep learning framework
- **scikit-learn**: Machine learning utilities
- **sentence-transformers**: Text embeddings
- **pandas**: Data manipulation
- **numpy**: Numerical computing

### Authentication & Security
- **PyJWT**: JSON Web Token implementation
- **bcrypt**: Password hashing

### Other
- **Pydantic**: Data validation
- **python-dotenv**: Environment variable management

## 📝 Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|---------|---------|
| `MONGO_URI` | MongoDB connection string | Yes | - |
| `MONGO_DB` | MongoDB database name | Yes | - |
| `JWT_SECRET_KEY` | Secret key for JWT tokens | Yes | - |
| `JWT_ALGORITHM` | JWT algorithm | No | HS256 |
| `JWT_EXPIRATION_HOURS` | JWT token expiration | No | 24 |
| `QDRANT_HOST` | Qdrant server URL | Yes | - |
| `QDRANT_API_KEY` | Qdrant API key | Yes | - |

## 🔍 Testing

### Test Health Endpoint

```bash
curl http://localhost:8000/health
```

### Test Authentication

```bash
# Signup
curl -X POST http://localhost:8000/auth/signup \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"test123","name":"Test User"}'

# Login
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"test123"}'
```

## 🤝 Contributing

This is a hackathon project. For contributions:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is developed for the Finlens AI Hackathon.

## 🆘 Troubleshooting

### Common Issues

1. **MongoDB Connection Error**
   - Ensure MongoDB is running
   - Check `MONGO_URI` in `.env`

2. **Qdrant Connection Error**
   - Verify Qdrant instance is accessible
   - Check `QDRANT_HOST` and `QDRANT_API_KEY`

3. **Model Training Fails**
   - Ensure training data has at least 2 categories
   - Check CSV format matches expected structure

4. **Import Errors**
   - Make sure virtual environment is activated
   - Run `pip install -r requirements.txt` again

5. **Port Already in Use**
   - Change port: `uvicorn app.main:app --port 8001`
   - Or kill process using port 8000

## 📞 Support

For issues or questions:
- Review API documentation at `/docs` endpoint
- Check logs for error messages

## 🎯 Hackathon Notes

### Quick Demo Setup

1. Start MongoDB and Qdrant
2. Create admin account: `POST /admin/auth/signup`
3. Train model: `POST /admin/train-model` with sample data
4. Create user account: `POST /auth/signup`
5. Upload transactions: `POST /transactions/upload-transactions`
6. View analytics: `GET /transactions/analytics`

### Key Features to Demo

- ✅ Automatic transaction categorization
- ✅ User feedback system
- ✅ Analytics dashboard
- ✅ CSV bulk upload
- ✅ Merchant knowledge base

---

**Built with ❤️ for Finlens AI Hackathon**

