# Use Ubuntu-based Node.js image for better compatibility with ONNX
FROM node:20-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for ONNX runtime
RUN apt-get update && apt-get install -y \
    libc6 \
    libgcc-s1 \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# Copy package files
COPY package.json ./

# Install dependencies
RUN npm install --omit=dev

# Copy server file
COPY server.js ./

# Create non-root user
RUN groupadd --gid 1001 nodejs && \
    useradd --uid 1001 --gid nodejs --shell /bin/bash --create-home nodejs

# Change ownership
RUN chown -R nodejs:nodejs /app

# Switch to non-root user
USER nodejs

# Expose port (Railway will set PORT env var)
EXPOSE 3000

# Set memory limits for Node.js
ENV NODE_OPTIONS="--max-old-space-size=1536"

# Start the server
CMD ["npm", "start"]