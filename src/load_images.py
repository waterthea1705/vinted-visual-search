from PIL import Image
import open_clip

# Load the pretrained OpenCLIP model and its image preprocessing steps
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k"
)

# Load the reference images
image_1 = Image.open("data/reference/ginger_jar_lamp_1.jpg")
image_2 = Image.open("data/reference/ginger_jar_lamp_2.jpg")

# Check ref images have loaded successfully
print(image_1)
print(image_2)

# Preprocess the reference images into tensors for the model
tensor_1 = preprocess(image_1)
tensor_2 = preprocess(image_2)

# Check the resulting tensor shapes
print(tensor_1.shape)
print(tensor_2.shape)

# Add a batch dimension so each tensor can be passed through the model
tensor_1 = tensor_1.unsqueeze(0)
tensor_2 = tensor_2.unsqueeze(0)

# Generate an embedding for each reference image
embedding_1 = model.encode_image(tensor_1)
embedding_2 = model.encode_image(tensor_2)

# Check the resulting embedding shapes
print(embedding_1.shape)
print(embedding_2.shape)

# Normalise the embeddings so they can be compared using cosine similarity
embedding_1 = embedding_1 / embedding_1.norm(dim=-1, keepdim=True)
embedding_2 = embedding_2 / embedding_2.norm(dim=-1, keepdim=True)

# Calculate the cosine similarity between the two reference images
similarity = (embedding_1 @ embedding_2.T).item()

print("Similarity:", similarity)
