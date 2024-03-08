import streamlit as st
import torch
from PIL import Image
from torchvision.transforms import transforms
from get_loader import get_loader
from model import CNNtoRNN

transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

train_loader, dataset = get_loader(
        root_folder='flickr8k/images',
        annotation_file='flickr8k/captions.txt',
        transform=transform,
        num_workers=2,
        batch_size=32
    )

# path to the trained model
MODEL_PATH = 'my_checkpoint.pth.tar'

# Function to load checkpoint
def load_checkpoint(model, optimizer, filename='checkpoint.pth.tar'):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    step = checkpoint['step']
    return model, optimizer, step

# Load trained model
model = CNNtoRNN(embed_size=256, hidden_size=1024, vocab_size=len(dataset.vocab), num_layers=2)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
model, optimizer, _ = load_checkpoint(model, optimizer, MODEL_PATH)
model.eval()

# Function to generate caption for the given image
def generate_caption(image, model, vocab):
    # Adding batch dimension
    image = transform(image).unsqueeze(0)

    caption = model.caption_image(image, vocab)
    return ' '.join(caption[1:-1])  # Excluding <SOS> and <EOS> tokens

# Streamlit app
def main():
    st.title("Image Captioning App")
    st.write("Upload an image and let the model generate a caption!")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        if st.button('Generate Caption'):
            _, dataset = get_loader('flickr8k/images/', annotation_file='flickr8k/captions.txt', transform=transform)
            vocab = dataset.vocab
            caption = generate_caption(image, model, vocab)
            st.write('**Caption:**', caption)

if __name__ == "__main__":
    main()
