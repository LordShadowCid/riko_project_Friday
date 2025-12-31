#!/bin/bash

# Riko Project - GPT-SoVITS Docker Setup Script
echo "🚀 Starting Riko Project with GPT-SoVITS Docker..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose is not available. Please install Docker Compose."
    exit 1
fi

# Create models directory if it doesn't exist
mkdir -p models

echo "📦 Starting GPT-SoVITS container..."

# Use docker compose if available, otherwise fall back to docker-compose
if docker compose version &> /dev/null; then
    docker compose up -d gpt-sovits
else
    docker-compose up -d gpt-sovits
fi

echo "⏳ Waiting for GPT-SoVITS to start..."
sleep 10

# Check if the service is running
if curl -f http://localhost:9880/health > /dev/null 2>&1; then
    echo "✅ GPT-SoVITS is running on http://localhost:9880"
else
    echo "⚠️  GPT-SoVITS is starting up... This may take a few minutes on first run."
    echo "   Check logs with: docker logs riko-gpt-sovits"
fi

echo ""
echo "🎯 Next steps:"
echo "1. Wait for GPT-SoVITS to fully start (check logs: docker logs riko-gpt-sovits)"
echo "2. Test the API: curl http://localhost:9880/health"
echo "3. Run your Riko chat: python server/main_chat.py"
echo ""
echo "🛑 To stop: docker compose down"