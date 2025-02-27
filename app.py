import streamlit as st
import torch
from tokenizers import Tokenizer
import torch.nn as nn

# Load the trained tokenizer
tokenizer = Tokenizer.from_file("bpe_tokenizer.json")

# Define the same Transformer model structure used during training
class CustomTransformer(nn.Module):
    def __init__(self, vocab_size=8000, embed_dim=128, hidden_dim=256, num_heads=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        return self.decoder(x)

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomTransformer().to(device)
model.load_state_dict(torch.load("transformer_model.pth", map_location=device))
model.eval()


# Streamlit UI
st.title("Pseudocode to C++ Converter")
st.markdown("Enter pseudocode below, and the model will generate the corresponding C++ code.")

pseudocode_input = st.text_area("Enter Pseudocode:", height=150)

if st.button("Convert to C++"):
    if pseudocode_input.strip():
        # Tokenize the input pseudocode
        input_tokens = tokenizer.encode(pseudocode_input).ids
        input_tensor = torch.tensor([input_tokens], dtype=torch.long).to(device)
        
        # Generate C++ code using the model
        with torch.no_grad():
            output_tokens = model(input_tensor)
        
        # Convert tokenized output back to text
        output_text = tokenizer.decode(output_tokens.argmax(dim=-1).tolist()[0])

        st.subheader("Generated C++ Code:")
        st.code(output_text, language="cpp")
    else:
        st.warning("Please enter pseudocode before generating.")
