import gdown
url = 'https://drive.google.com/uc?id=12nfzhzHt5Eo2jkKQE2-r171KaL_ASYm5'
output = 'models/dnn_model.pth'
gdown.download(url, output, quiet=False)