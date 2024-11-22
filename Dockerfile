
FROM bitnami/spark:latest

# Switch to root to install system packages
USER root

# Install Python, pip, and system-level dependencies
RUN apt-get update && apt-get install -y python3-pip && \
    pip3 install numpy matplotlib pandas librosa scipy streamlit && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Switch back to the default non-root user
USER 1001
