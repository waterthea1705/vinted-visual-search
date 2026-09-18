from pathlib import Path

from PIL import Image
import open_clip
import torch

# Load the pretrained OpenCLIP model and its image preprocessing steps
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k"
)

# Load and preprocess the reference images
reference_paths = [
    Path("data/reference/ginger_jar_lamp_1.jpg"),
    Path("data/reference/ginger_jar_lamp_2.jpg")
]

reference_tensors = []

for path in reference_paths:
    image = Image.open(path)
    tensor = preprocess(image)
    reference_tensors.append(tensor)

# Combine the reference tensors into one batch
reference_batch = torch.stack(reference_tensors)

print(reference_batch.shape)

# -------------------------------------------------
# Generate an embedding for each reference image
with torch.no_grad():
    reference_embeddings = model.encode_image(reference_batch)

# Normalise each reference embedding
reference_embeddings = reference_embeddings / reference_embeddings.norm(
    dim=-1, keepdim=True
)

# Average the two embeddings to create one combined reference
reference_embedding = reference_embeddings.mean(dim=0, keepdim=True)

# Normalise the combined reference embedding
reference_embedding = reference_embedding / reference_embedding.norm(
    dim=-1, keepdim=True
)

print(reference_embedding.shape)

# -------------------------------------------------
# Find all candidate lamp images
candidate_paths = list(Path("data/candidates").glob("*.png"))

# Load and preprocess each candidate image
candidate_tensors = []

for path in candidate_paths:
    image = Image.open(path)
    tensor = preprocess(image)
    candidate_tensors.append(tensor)

# Combine all candidate tensors into one batch
candidate_batch = torch.stack(candidate_tensors)

print(candidate_batch.shape)

# -------------------------------------------------
# Generate embeddings for all candidate images
with torch.no_grad():
    candidate_embeddings = model.encode_image(candidate_batch)

# Normalise the candidate embeddings
candidate_embeddings = candidate_embeddings / candidate_embeddings.norm(
    dim=-1, keepdim=True
)

print(candidate_embeddings.shape)

# -------------------------------------------------
# Calculate the similarity between the reference and every candidate
similarities = reference_embedding @ candidate_embeddings.T

# Convert the similarity scores into a simple one-dimensional tensor
similarities = similarities.squeeze(0)

# Rank the candidates from highest to lowest similarity
ranked_indices = torch.argsort(similarities, descending=True)

# Print the ranked search results
for rank, index in enumerate(ranked_indices, start=1):
    path = candidate_paths[index]
    score = similarities[index].item()

    print(f"{rank}. {path.name}: {score:.3f}")
