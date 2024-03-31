# Use an official Ubuntu 22.04 base image
FROM ubuntu:22.04

# Install necessary utilities including wget, git, zsh, vim, tree, htop, curl, and tmux
RUN apt-get update && apt-get install -y wget git zsh vim tree htop curl tmux fzf

# # Set default shell to Zsh
SHELL ["/bin/zsh", "-c"]
RUN chsh -s $(which zsh)

# Install Oh My Zsh with extensions
RUN sh -c "$(wget https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh -O -)"
# Add any specific Oh My Zsh extensions here
RUN git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions
RUN git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting
RUN git clone https://github.com/marlonrichert/zsh-autocomplete ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autocomplete
# Activate extensions
RUN sed -i 's/plugins=(.*)/plugins=(git zsh-autocomplete zsh-autosuggestions zsh-syntax-highlighting)/' ~/.zshrc
RUN sed -i 's/robbyrussell/takashiyoshida/g' ~/.zshrc



# Install Miniconda for Python 3.9 and PyTorch
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p $HOME/miniconda && \
    rm ~/miniconda.sh
ENV PATH="/root/miniconda/bin:${PATH}"
RUN conda install -y python=3.9 && \
    conda install -y pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch

# Install NVM, Node.js, and npm
ENV NVM_DIR /root/.nvm
ENV NODE_VERSION lts/*
RUN wget -qO- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.1/install.sh | bash && \
    . $NVM_DIR/nvm.sh && \
    nvm install $NODE_VERSION && \
    nvm use $NODE_VERSION && \
    nvm alias default $NODE_VERSION
ENV PATH $NVM_DIR/versions/node/v`node -v`/bin:$PATH

# Install neovim and astrovim
RUN cd ~ && wget https://github.com/neovim/neovim/releases/download/v0.9.5/nvim-linux64.tar.gz 
RUN tar xzvf ~/nvim-linux64.tar.gz -C ~ && rm ~/nvim-linux64.tar.gz
# source the binary
RUN echo "export PATH=~/nvim-linux64/bin:$PATH" >> ~/.zshrc
# install astrovim
RUN git clone --depth 1 https://github.com/AstroNvim/AstroNvim ~/.config/nvim

RUN mkdir ~/DSC-180B-Hardware-Acceleration

# TODO: write the cmd for installing NVIDIA CUDA 11.8.0

# Set working directory
WORKDIR /root/home/DSC-180B-Hardware-Acceleration

# Default command to run when starting the container
CMD ["zsh"]
