from torch import nn
import torch
import os
import json

class PoolingConcat(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.dimension = dimension
        self.dense = nn.Sequential(
              nn.Linear(dimension*2, dimension)
            )

    def forward(self, features: dict[str, torch.Tensor], **kwargs) -> dict   [str, torch.Tensor]:

        token_embeddings = features["token_embeddings"][:,1:,:]
        attention_mask = features["attention_mask"].unsqueeze(-1)[:,1:]
        token_embeddings = token_embeddings * attention_mask
        mean_embeddings = token_embeddings.sum(1) / attention_mask.sum(1)

        cls_embeddings = features["token_embeddings"][:,0,:]

        pooled = torch.concat([mean_embeddings, cls_embeddings], dim=-1)

        features["sentence_embedding"] = self.dense(pooled)

        return features

    def get_config_dict(self) -> dict[str, float]:
        return {"dimension": self.dimension}

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension

    def save(self, save_dir: str, **kwargs) -> None:
        with open(os.path.join(save_dir, "config.json"), "w") as fOut:
            json.dump(self.get_config_dict(), fOut, indent=4)

    def load(load_dir: str, **kwargs):
        with open(os.path.join(load_dir, "config.json")) as fIn:
            config = json.load(fIn)

        return PoolingConcat(**config)