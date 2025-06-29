import numpy as np
import tritonclient.grpc as grpcclient
from typing import List, Union, Any
from pydantic import BaseModel
from tritonclient.utils import np_to_triton_dtype

from langchain_core.embeddings import Embeddings


class TritonEmbeddings(BaseModel, Embeddings):
    """
    A client for getting text embeddings from a Triton Inference Server.
    """

    url: str = "localhost:8001"
    model_name: str

    input_name: str
    output_name: str

    def model_post_init(self, context: Any) -> None:
        self._client = grpcclient.InferenceServerClient(url=self.url)

        # Verify server is ready
        if not self._client.is_server_ready():
            raise ConnectionError(f"Triton server at {self.url} is not ready")

        # Verify model is ready
        if not self._client.is_model_ready(self.model_name):
            raise ValueError(f"Model {self.model_name} is not ready")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Create input object for text strings
        text_array = np.array(texts, dtype=object).reshape(-1)
        input_text = grpcclient.InferInput(
            self.input_name,
            text_array.shape,
            np_to_triton_dtype(text_array.dtype),
        )
        input_text.set_data_from_numpy(text_array)

        # Create output object for embeddings
        output_emb = grpcclient.InferRequestedOutput(self.output_name)

        # Send request to server - Triton will handle batching automatically
        response = self._client.infer(
            model_name=self.model_name,
            inputs=[input_text],
            outputs=[output_emb],
        )

        # Get embeddings from response
        embeddings: np.ndarray[np.floating] = response.as_numpy(self.output_name)  # type: ignore
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

    def __del__(self):
        """Clean up Triton client when object is deleted."""
        if hasattr(self, "_client"):
            self._client.close()
