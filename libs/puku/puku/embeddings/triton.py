import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype
from typing import List, Union

from langchain_core.embeddings import Embeddings
from puku_core._api.experimental_decorator import experimental


@experimental(addendum="P.S in-process")
class TritonEmbeddings(Embeddings):
    """
    A client for getting text embeddings from a Triton Inference Server.

    Args:
        url (str): URL of the Triton server (e.g., 'localhost:8001')
        model_name (str): Name of the embedding model in Triton
        timeout (float): Timeout in seconds for server communication
        verbose (bool): Whether to print verbose output
    """

    def __init__(
        self,
        url: str = "localhost:8001",
        model_name: str = "embedding_model",
        timeout: float = 10.0,
        verbose: bool = False,
    ):
        self.url = url
        self.model_name = model_name
        self.timeout = timeout
        self.verbose = verbose

        # Initialize Triton client
        self.client = grpcclient.InferenceServerClient(url=url, verbose=verbose)

        # Verify server is ready
        if not self.client.is_server_ready():
            raise ConnectionError(f"Triton server at {url} is not ready")

        # Verify model is ready
        if not self.client.is_model_ready(model_name):
            raise ValueError(f"Model {model_name} is not ready")

    def get_embeddings(
        self, texts: Union[str, List[str]], batch_size: int = 32
    ) -> np.ndarray:
        """
        Get embeddings for one or more text strings.

        Args:
            texts: A single string or list of strings to embed
            batch_size: Number of texts to process in each batch

        Returns:
            numpy.ndarray: Embeddings with shape (num_texts, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]

        # Process in batches if there are many texts
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_embeddings = self._get_batch_embeddings(batch_texts)
            all_embeddings.append(batch_embeddings)

        return np.vstack(all_embeddings)

    def _get_batch_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for a single batch of texts."""
        # Create input object for text strings
        text_array = np.array(texts, dtype=object).reshape(-1, 1)
        input_text = grpcclient.InferInput(
            "TEXT",  # This should match your model's input name
            text_array.shape,
            np_to_triton_dtype(text_array.dtype),
        )
        input_text.set_data_from_numpy(text_array)

        # Create output object for embeddings
        output_emb = grpcclient.InferRequestedOutput(
            "EMBEDDING_OUTPUT"  # This should match your model's output name
        )

        # Send request to server
        response = self.client.infer(
            model_name=self.model_name,
            inputs=[input_text],
            outputs=[output_emb],
            timeout=self.timeout,
        )

        # Get embeddings from response
        embeddings = response.as_numpy("EMBEDDING_OUTPUT")
        return embeddings

    def __del__(self):
        """Clean up Triton client when object is deleted."""
        if hasattr(self, "client"):
            self.client.close()
