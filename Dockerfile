FROM continuumio/miniconda3

# Set working directory
WORKDIR /image-retrieval

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    wget \
    git

# Install Python packages using Miniconda
RUN conda install -y python=3.10 numpy scipy matplotlib pandas jupyterlab opencv pytorch torchvision torchaudio faiss-cpu streamlit -c pytorch -c conda-forge

# Expose ports
EXPOSE 9999 8501

# Default command to launch JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=9999", "--no-browser", "--allow-root"]