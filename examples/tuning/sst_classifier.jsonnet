local embedding_dim = std.parseJson(std.extVar('embedding_dim'));
local hidden_dim = std.parseJson(std.extVar('hidden_dim'));
local lr = std.parseJson(std.extVar('lr'));

{
  "dataset_reader": {
    "type": "sst_tokens"
  },
  "train_data_path": "https://s3.amazonaws.com/realworldnlpbook/data/stanfordSentimentTreebank/trees/train.txt",
  "validation_data_path": "https://s3.amazonaws.com/realworldnlpbook/data/stanfordSentimentTreebank/trees/dev.txt",

  "model": {
    "type": "lstm_classifier",

    "embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": embedding_dim
        }
      }
    },

    "encoder": {
      "type": "lstm",
      "input_size": embedding_dim,
      "hidden_size": hidden_dim
    }
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "sorting_keys": ["tokens"],
      "padding_noise": 0.1,
      "batch_size" : 32
    }
  },
  "trainer": {
    "optimizer": {
      "type": "adam",
      "lr": lr,
    },
    "validation_metric": "+accuracy",
    "num_epochs": 20,
    "patience": 10,
    "cuda_device": 0
  }
}
