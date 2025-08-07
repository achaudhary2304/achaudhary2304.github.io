---
title: "Hangman Challenge"
excerpt: "A simple strategy for training a bi-LSTM for playing Hangman"
collection: portfolio
---
  

# Hangman Challenge - LSTM Model Implementation
## My Approach and Code Explanation



## Overview

This is the writeup for my attempt at the Hangman Challenge. To run the file you have to create an environment with torch. My strategy initially was to train using a RNN and then move on to LSTM. On the way I figured out ways to improve the accuracy more and more and make the whole thing better. The `train.py` script contains each and every thing required for the whole thing.




## Data Preparation

I create the Dataset to train the LSTM model using the following functions.

### Character Mapping
First I create a char map to tokenize each and every character that can be present, from a to z and the PAD token and the _ token. So I have a char map of 28 things:

```python
def mapping_chars():
    
    char_map = {'_': 27, 'PAD': 0}
    count = 1 
    for char in list(string.ascii_lowercase):
        char_map[char] = count
        count += 1
    return char_map
```

This gives me:
- PAD: 0 (for padding shorter words)
- a-z: 1-26 
- _: 27 (for masked letters)

### Creating Training Pairs
Then I read all the words in the txt file and create pairs out of them. I create pairs such as the actual original word, then the blank word and also an already guessed list.The limit was enforced because if not then the number of pairs grew at a exponential rate.
Right now this only creates single _ words.

```python
def create_pairs_limited(word: str, max_pairs: int = 10) -> List[Tuple[str, str]]:
    
    unique_letters = list(set(word))
    pairs = []
    
    for letter in unique_letters:
        masked_word = word.replace(letter, '_')
        pairs.append((masked_word, word))
        if len(pairs) >= max_pairs:
            break
```

For example, if I have the word "python":
- Original: "python"
- Masked versions: "p_thon", "py_hon", "pyt_on", etc.
- Each pair: (masked_word, original_word)

### Contextual Simulation
The reason I did was that earlier it used to always predict a and e and other high probabilty letters with the highest proab. but I had to filter all the tested letters so as to take the least one 
This part is interesting so what I do is that I try to simulate the Hangman game and create true and false chances. All this is done to give the model context of what it had already tried so that it would not again give the same suggestion with high probability.

```python
def process_words_contextual(words: List[str], char_map: Dict[str, int], max_pairs_per_word: int = 5):
    
    word_letters = set(original_word)
    revealed_letters = set(masked_word.replace('_', ''))
    
    
    plausible_incorrect_guesses = [l for l in STANDARD_LETTER_FREQ if l not in word_letters]
    
    
    already_guessed = list(revealed_letters)
    
    num_incorrect_to_add = random.randint(0, 6)
    incorrect_guesses = random.sample(plausible_incorrect_guesses, k=min(num_incorrect_to_add, len(plausible_incorrect_guesses)))
    
    
    all_guessed = already_guessed + incorrect_guesses
```

So for "python" with mask "p_th__", I might have:
- Correct guesses: ['p', 't', 'h'] (already revealed)
- Wrong guesses: ['e', 'a', 'r'] (random common letters not in word)
- Full context: ['p', 't', 'h', 'e', 'a', 'r']

This way the model learns what letters have already been tried and won't suggest them again.



## Model Architecture

### The LSTM Model
I use a bi-LSTM model that takes two inputs:

```python
class ContextualWordPredictionModel(nn.Module):
    def __init__(self, vocab_size=28, input_len=20, output_len=26, hidden_size=256, embedding_dim=64):
        super(ContextualWordPredictionModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True, num_layers=2, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size * 2 + 26, output_len) # *2 for bidirectional
        self.sigmoid = nn.Sigmoid()
```

**Input 1:** The masked word (like "p_th__") converted to numbers
**Input 2:** A binary vector of 26 values showing which letters were already guessed

The model combines both inputs:
- LSTM processes the masked word pattern
- Binary vector shows which letters are off-limits
- Final layer combines both to predict next letter

### Forward Pass
```python
def forward(self, x, guessed_mask):
    embedded = self.embedding(x)
    lstm_out, (hidden, _) = self.lstm(embedded)
    lstm_out = lstm_out[:, -1, :]
    combined_out = torch.cat((lstm_out, guessed_mask), dim=1)
    output = self.dropout(combined_out)
    output = self.fc(output)
    output = self.sigmoid(output)
    return output
```

The LSTM gives me 512 features (256*2 for bidirectional), plus 26 features for guessed letters = 538 total features going into the final layer.



## Encoding Functions

### Input Encoding
```python
def encode_input(word: str, char_map: Dict[str, int]) -> torch.Tensor:
    """Encodes the masked word into a tensor."""
    input_len = 20
    input_tensor = torch.zeros(input_len, dtype=torch.long)
    start_pos = max(0, input_len - len(word))
    
    for i, char in enumerate(word):
        if start_pos + i < input_len:
            input_tensor[start_pos + i] = char_map.get(char, char_map['PAD'])
    return input_tensor
```

This converts "p_th__" to numbers like [0,0,0,0,0,0,0,0,0,0,0,0,0,0,16,27,20,8,27,27]

### Guessed Letters Encoding
```python
def encode_guessed_letters(guessed: List[str], char_map: Dict[str, int]) -> torch.Tensor:
    """Encodes the list of guessed letters into a binary tensor."""
    tensor = torch.zeros(26, dtype=torch.float)
    for letter in guessed:
        if letter in char_map and letter not in ['_', 'PAD']:
            tensor[char_map[letter] - 1] = 1
    return tensor
```

This converts ['p', 't', 'h', 'e', 'a'] to a binary vector where positions for these letters are 1, rest are 0.

### Output Encoding
```python
def encode_output(original_word: str, masked_word: str, char_map: Dict[str, int]) -> torch.Tensor:
    """Encodes the target letter(s) into a probability distribution."""
    missing_letter_counts = {}
    for i in range(len(original_word)):
        if masked_word[i] == '_' and original_word[i] != '_':
            char = original_word[i]
            if char in char_map and char not in ['PAD', '_']:
                missing_letter_counts[char] = missing_letter_counts.get(char, 0) + 1

    # Convert to probability distribution
    for char, count in missing_letter_counts.items():
        idx = char_map[char] - 1
        if 0 <= idx < output_len:
            output_tensor[idx] = count
    
    total_missing_letters = torch.sum(output_tensor)
    if total_missing_letters > 0:
        output_tensor = output_tensor / total_missing_letters
```

For "python" vs "p_th__", the missing letters are 'y', 'o', 'n', so the output has high probability for these positions.



## Training Process

### Training Configuration
```python
def main():
    DATA_FILE = "words_250000_train.txt"
    MAX_WORDS = 250000       # Use all the data
    MAX_PAIRS_PER_WORD = 15
    BATCH_SIZE = 512
    NUM_EPOCHS = 15          # Increased epochs for deeper learning
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-5
```

I use:
- 250,000 words from the training file
- Up to 15 pairs per word (so ~3.75M training examples)
- Batch size of 512
- 15 epochs
- Adam optimizer
- BCELoss since it's a multi-label classification

### Training Loop
```python
def train_contextual(dataloader, model, optimizer, criterion, num_epochs=2, device='cpu'):
    model.train()
    model.to(device)
    
    for epoch in range(num_epochs):
        for batch_idx, (X_batch, G_batch, Y_batch) in enumerate(tqdm(dataloader)):
            X_batch, G_batch, Y_batch = X_batch.to(device), G_batch.to(device), Y_batch.float().to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch, G_batch)
            loss = criterion(outputs, Y_batch)
            
            loss.backward()
            optimizer.step()
```

Standard PyTorch training loop with error handling and progress tracking.





## Environment Setup

To run this code:
```bash
pip install torch polars tqdm
python train.py
```

The script will:
1. Load the word dataset
2. Generate training pairs with context
3. Train the LSTM model
4. Save the model as 'contextual_word_model.pt'



## Results

 The contextual approach significantly improves performance compared to basic frequency-based methods by incorporating game state awareness and avoiding repeated incorrect guesses.trex
