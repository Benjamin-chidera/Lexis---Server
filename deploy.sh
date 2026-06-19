#!/bin/bash
set -e

SERVER_IP="2.24.131.197"
USER="root"

echo "Deploying to $USER@$SERVER_IP..."

# 1. Stop and disable Dokploy containers to free up port 80 and 443
echo "Stopping Dokploy services (if running)..."
ssh $USER@$SERVER_IP << 'EOF'
  docker stop dokploy-traefik dokploy || true
  docker rm dokploy-traefik dokploy || true
  # Optionally stop any other dokploy containers if they conflict
  mkdir -p /root/legal-assistant/server
EOF

# 2. Copy the necessary files over
echo "Copying files to server..."
scp docker-compose.yml Dockerfile pyproject.toml requirements.txt worker.py main.py database.py models.py auth.py run_worker.py cloudinary_client.py .env $USER@$SERVER_IP:/root/legal-assistant/server/
scp -r ai routes $USER@$SERVER_IP:/root/legal-assistant/server/

# 3. Build and start the docker-compose stack
echo "Building and starting Docker Compose stack..."
ssh $USER@$SERVER_IP << 'EOF'
  cd /root/legal-assistant/server
  docker compose down || true
  docker compose up -d --build
  echo "✅ Deployment successful! API is running."
  docker compose ps
EOF

echo "All done! Your server should now be accessible at https://lexis-api.discoverbenix.com"
