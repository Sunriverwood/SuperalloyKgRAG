# Copyright 2025 SUNRIVERWOOD
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Reasoning Query Handler

Integrates graph reasoning capabilities with the existing query system.
Provides a unified interface for query-aware graph reasoning.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml
import torch
import numpy as np
from openai import OpenAI
import torch.cuda

# --- 项目根目录定义 (必须在导入 core 之前) ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# 将项目根目录添加到 Python 路径，以便可以导入 core 模块
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.reasoning.data_loader import GraphReasoningDataLoader, GraphData
from core.reasoning.models.rgat import QueryAwareRGAT
from core.reasoning.training.trainer import GraphReasoningTrainer, QueryEntityMatcher
from core.reasoning.inference.reasoner import GraphReasoner


def setup_logging(config: Dict[str, Any]):
    """根据配置文件设置日志记录器"""
    log_config = config.get("logging", {})
    level = getattr(logging, log_config.get("level", "INFO").upper(), logging.INFO)
    relative_log_path = log_config.get("log_file", "logs/superalloyKgRAG.log")
    log_file = PROJECT_ROOT / relative_log_path

    log_file.parent.mkdir(exist_ok=True, parents=True)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='a', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logging.info("推理查询日志记录器设置完成")


def load_config(settings_filename: str = "settings.yaml") -> Dict[str, Any]:
    """加载YAML配置文件"""
    config_path = PROJECT_ROOT / "config" / settings_filename
    logging.info(f"正在从 {config_path} 加载配置...")
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件 {config_path} 未找到！")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    logging.info("配置加载成功。")
    return config


class ReasoningQueryHandler:
    """
    Query handler with graph reasoning capabilities.

    Workflow:
    1. Query encoding (using text encoder)
    2. Node retrieval (initial candidates)
    3. Graph reasoning (PPR propagation + path extraction)
    4. Answer generation (LLM synthesis with reasoning paths)
    """

    def __init__(self, config: Dict[str, Any], load_trained_model: bool = True, shared_graph_data: Dict = None):
        """
        Args:
            config: Configuration dictionary
            load_trained_model: Whether to load pre-trained model (False for training mode)
            shared_graph_data: Optional pre-loaded graph data to avoid redundant loading (memory optimization)
        """
        self.config = config
        self.reasoning_config = config.get("reasoning", {})

        # Setup API client for text encoding and generation
        self.api_key = os.getenv("QWEN_API_KEY")
        if not self.api_key:
            logging.warning("未找到 QWEN_API_KEY 环境变量")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

        self.embedding_model_name = config["embedding"]["model"]
        self.generation_model_name = config["query"]["generation_model"]
        self.temperature = config["query"]["temperature"]

        # Device setup - CRITICAL: Load data to CPU first to avoid OOM
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Target device: {self.device}")

        # MEMORY OPTIMIZATION: Always load graph data to CPU first
        # Only move necessary parts to GPU during forward pass
        data_device = 'cpu'
        logging.info(f"🔧 [Memory Optimization] Loading graph data to CPU first to prevent OOM")

        # Load graph data - use shared data if provided (memory optimization)
        if shared_graph_data is not None:
            logging.info("💡 [内存优化] 使用共享的图数据，避免重复加载")
            self.graph_data = shared_graph_data
            self.data_loader = None  # 不需要 data_loader
        else:
            logging.info("Loading graph data for reasoning...")
            self.data_loader = GraphReasoningDataLoader(config)
            # CRITICAL: Load to CPU first
            self.graph_data = self.data_loader.load(device=data_device)

            # Log memory usage
            if self.device.type == 'cuda':
                allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)  # GB
                reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)  # GB
                logging.info(
                    f"📊 GPU Memory after graph loading: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

        # Load chunk ID to source info mapping
        logging.info("Loading chunk ID to source mapping...")
        self.chunk_id_map = self._load_chunk_id_map()

        # Load text units for source text snippets
        logging.info("Loading text units for source reference...")
        self.text_units = self._load_text_units()

        # Initialize or load models
        if load_trained_model:
            self._load_trained_models()
        else:
            self._initialize_models()

        # Create reasoner
        self.reasoner = GraphReasoner(
            config=config,
            graph_data=self.graph_data,
            gnn_model=self.gnn,
            query_matcher=self.query_matcher,
            device=self.device
        )

        logging.info("ReasoningQueryHandler initialized successfully")

    def _load_chunk_id_map(self) -> Dict[str, Dict[str, Any]]:
        """
        Load chunk ID to source info mapping (source_filename, pages, blocks).
        Loads from all units files: text, abstract, image, table.

        Returns:
            Dictionary mapping chunk_id to source metadata
        """
        chunk_map = {}

        # Define all units files to load with their chunk type
        units_files = [
            ("text_units.jsonl", "text"),
            ("abstract_units.jsonl", "abstract"),
            ("image_units.jsonl", "image"),
            ("table_units.jsonl", "table")
        ]

        chunks_dir = PROJECT_ROOT / "data" / "chunks"
        total_loaded = 0

        for filename, chunk_type in units_files:
            units_path = chunks_dir / filename

            try:
                if not units_path.exists():
                    logging.debug(f"⚠ {filename} not found, skipping")
                    continue

                logging.info(f"Loading {chunk_type} units from {filename}...")
                file_count = 0

                with open(units_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            unit = json.loads(line)
                            chunk_id = unit.get("id")
                            metadata = unit.get("metadata", {})

                            if chunk_id:
                                # For abstract type, use different fields
                                if chunk_type == "abstract":
                                    chunk_map[chunk_id] = {
                                        "source_filename": metadata.get("source_filename", "unknown"),
                                        "journal": metadata.get("journal", "unknown"),
                                        "year": metadata.get("year", "unknown")
                                    }
                                else:
                                    # For text, image, table types, use pages and blocks
                                    chunk_map[chunk_id] = {
                                        "source_filename": metadata.get("source_filename", "unknown"),
                                        "pages": metadata.get("pages", []),
                                        "blocks": metadata.get("blocks", [])
                                    }
                                file_count += 1

                logging.info(f"  ✓ Loaded {file_count} mappings from {filename}")
                total_loaded += file_count

            except Exception as e:
                logging.warning(f"⚠ Error loading {filename}: {e}")
                continue

        logging.info(f"✓ Total: loaded {total_loaded} chunk ID mappings")

        if total_loaded == 0:
            logging.warning("⚠ No chunk ID mappings loaded, source references will not be available")

        return chunk_map

    def _load_text_units(self) -> Dict[str, str]:
        """
        Load text units to create chunk_id -> text mapping for source reference.

        Returns:
            Dictionary mapping chunk_id to original text content
        """
        text_units_path = PROJECT_ROOT / self.config["embedding"]["input_text_units_path"]
        text_map = {}
        try:
            logging.info(f"Loading text units from: {text_units_path}")
            with open(text_units_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        unit = json.loads(line)
                        chunk_id = unit.get("id")
                        text = unit.get("text", "")
                        if chunk_id and text:
                            text_map[chunk_id] = text
            logging.info(f"✓ Loaded {len(text_map)} text units for source reference")
            return text_map
        except FileNotFoundError:
            logging.warning(f"⚠ Text units file not found: {text_units_path}")
            return {}
        except Exception as e:
            logging.error(f"✗ Error loading text units: {e}", exc_info=True)
            return {}

    def _format_source_reference(self, chunk_ids: List[str]) -> str:
        """
        Format chunk IDs as human-readable source references.
        Format:

        Args:
            chunk_ids: List of chunk IDs

        Returns:
            Formatted source reference string
        """
        if not chunk_ids or not self.chunk_id_map:
            return ""

        def extract_base_chunk_id(chunk_id_raw: str) -> str:
            """Extract base chunk ID (remove suffixes like _table_0)."""
            import re
            match = re.match(r'^(chunk-[a-f0-9]+)', chunk_id_raw)
            return match.group(1) if match else chunk_id_raw

        def merge_pages(page_list: List[int]) -> str:
            """Merge consecutive pages into range format."""
            if not page_list:
                return ""
            unique_pages = sorted(set(page_list))
            if len(unique_pages) == 1:
                return f"Page {unique_pages[0]}"
            # Check if consecutive
            is_consecutive = all(unique_pages[i + 1] - unique_pages[i] == 1
                                 for i in range(len(unique_pages) - 1))
            if is_consecutive:
                return f"Page {unique_pages[0]}~{unique_pages[-1]}"
            elif len(unique_pages) <= 3:
                return f"Pages {', '.join(map(str, unique_pages))}"
            else:
                return f"Pages {unique_pages[0]}~{unique_pages[-1]}"

        def merge_blocks(block_list: List[str]) -> str:
            """Merge consecutive blocks into range format."""
            if not block_list:
                return ""
            unique_blocks = sorted(set(block_list))
            if len(unique_blocks) == 1:
                return unique_blocks[0]
            # Try to detect if blocks are consecutive numbers
            try:
                import re
                block_nums = []
                for block in unique_blocks:
                    match = re.match(r'(Block|block)_(\d+)', block)
                    if match:
                        block_nums.append(int(match.group(2)))
                    else:
                        block_nums = []
                        break
                if block_nums and len(block_nums) == len(unique_blocks):
                    is_consecutive = all(block_nums[i + 1] - block_nums[i] == 1
                                         for i in range(len(block_nums) - 1))
                    if is_consecutive:
                        return f"Block {unique_blocks[0]}~{unique_blocks[-1]}"
            except Exception:
                pass
            if len(unique_blocks) <= 3:
                return f"Blocks {', '.join(unique_blocks)}"
            else:
                return f"Blocks {unique_blocks[0]}~{unique_blocks[-1]}"

        # Group by source file
        sources_by_file: Dict[str, Dict[str, List]] = {}
        abstract_sources = []  # Separate handling for abstract type

        for chunk_id_raw in chunk_ids:
            chunk_id = extract_base_chunk_id(chunk_id_raw)
            if chunk_id in self.chunk_id_map:
                source_info = self.chunk_id_map[chunk_id]
                chunk_type = source_info.get("chunk_type", "text")
                filename = source_info.get("source_filename", "unknown").replace('.json', '')

                # For abstract type, use different format
                if chunk_type == "abstract":
                    journal = source_info.get("journal", "unknown")
                    year = source_info.get("year", "unknown")
                    abstract_sources.append(f"{filename}, {journal}, {year}")
                else:
                    # For text/image/table types, use pages and blocks
                    pages = source_info.get("pages", [])
                    blocks = source_info.get("blocks", [])

                    if filename not in sources_by_file:
                        sources_by_file[filename] = {"pages": [], "blocks": []}
                    sources_by_file[filename]["pages"].extend(pages)
                    sources_by_file[filename]["blocks"].extend(blocks)

        # Format source parts
        source_parts = []

        # Add abstract sources first
        if abstract_sources:
            source_parts.extend(abstract_sources)

        # Add other type sources
        for filename, info in sources_by_file.items():
            if filename == "unknown":
                continue
            page_str = merge_pages(info["pages"])
            block_str = merge_blocks(info["blocks"])
            source_parts.append(f"{filename} {page_str} {block_str}")

        if len(source_parts) == 0:
            return "[source: unknown]"
        elif len(source_parts) == 1:
            return f"[source: {source_parts[0]}]"
        else:
            return f"[source: {'; '.join(source_parts)}]"

    def _initialize_models(self):
        """Initialize models for training"""
        model_config = self.reasoning_config.get('model', {})

        self.gnn = QueryAwareRGAT(
            input_dim=self.graph_data.embed_dim,
            hidden_dim=model_config.get('hidden_dim', 256),
            output_dim=model_config.get('hidden_dim', 256),
            num_layers=model_config.get('num_layers', 3),
            num_heads=model_config.get('num_heads', 4),
            dropout=model_config.get('dropout', 0.1),
            use_edge_weights=model_config.get('use_edge_weights', True),
            query_dim=self.graph_data.embed_dim,
            edge_type_dim=self.graph_data.embed_dim  # Edge type embeddings stay in original dimension
        ).to(self.device)

        self.query_matcher = QueryEntityMatcher(
            query_dim=self.graph_data.embed_dim,
            entity_dim=model_config.get('hidden_dim', 256),
            hidden_dim=model_config.get('hidden_dim', 256)
        ).to(self.device)

        logging.info("Models initialized for training")

    def _load_trained_models(self):
        """Load pre-trained models"""
        model_path = PROJECT_ROOT / self.reasoning_config.get('output', {}).get('model_path', 'data/reasoning/model.pt')

        if not model_path.exists():
            logging.warning(f"Trained model not found at {model_path}, initializing new models")
            self._initialize_models()
            return

        logging.info(f"Loading trained model from {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device)

        # Initialize models with same architecture
        self._initialize_models()

        # Load state dicts
        self.gnn.load_state_dict(checkpoint['gnn_state_dict'])
        self.query_matcher.load_state_dict(checkpoint['query_matcher_state_dict'])

        # Set to eval mode
        self.gnn.eval()
        self.query_matcher.eval()

        logging.info(f"Loaded trained model from epoch {checkpoint.get('epoch', 'unknown')}")

    def encode_query(self, query_text: str) -> np.ndarray:
        """
        Encode query text using embedding model.

        Args:
            query_text: Natural language query

        Returns:
            Query embedding vector
        """
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model_name,
                input=query_text,
                dimensions=self.graph_data.embed_dim
            )

            embedding = np.array(response.data[0].embedding)
            return embedding

        except Exception as e:
            logging.error(f"Query encoding failed: {e}")
            # Fallback to zero vector
            return np.zeros(self.graph_data.embed_dim)

    def generate_answer(self, query: str, reasoning_results: Dict[str, Any],
                        strict_mode: bool = True) -> str:
        """
        Generate final answer using LLM with reasoning context.

        Args:
            query: Original query
            reasoning_results: Results from graph reasoning
            strict_mode: If True, strictly prohibit LLM from using its own knowledge

        Returns:
            Generated answer
        """
        # Format reasoning paths as context
        context_parts = []

        # Add top nodes with source references
        context_parts.append("## Relevant Entities:")
        relevant_chunks = set()
        for node_info in reasoning_results['top_nodes'][:5]:
            node_id = node_info['id']
            # Get chunk_id from graph node
            if node_id in self.graph_data.G.nodes:
                node_data = self.graph_data.G.nodes[node_id]
                chunk_ids = node_data.get('chunk_id') or node_data.get('text_unit_ids', [])
                if isinstance(chunk_ids, str):
                    chunk_ids = [chunk_ids]
                elif chunk_ids is None:
                    chunk_ids = []

                if chunk_ids:
                    source_ref = self._format_source_reference(chunk_ids)
                    context_parts.append(f"- {node_info['name']} (relevance: {node_info['score']:.3f}) {source_ref}")
                    relevant_chunks.update(chunk_ids)
                else:
                    context_parts.append(f"- {node_info['name']} (relevance: {node_info['score']:.3f})")
            else:
                context_parts.append(f"- {node_info['name']} (relevance: {node_info['score']:.3f})")

        # Add reasoning paths with source references
        if reasoning_results['paths']:
            context_parts.append("\n## Reasoning Paths:")
            for i, path_info in enumerate(reasoning_results['paths'][:3], 1):
                context_parts.append(f"\nPath {i} (confidence: {path_info['score']:.4f}):")
                context_parts.append(path_info['explanation'])

                # Collect chunk_ids from path nodes and add source reference
                path_chunks = []
                for node_id in path_info['path']:
                    if node_id in self.graph_data.G.nodes:
                        node_data = self.graph_data.G.nodes[node_id]
                        chunk_ids = node_data.get('chunk_id') or node_data.get('text_unit_ids', [])
                        if isinstance(chunk_ids, str):
                            path_chunks.append(chunk_ids)
                            relevant_chunks.add(chunk_ids)
                        elif isinstance(chunk_ids, list):
                            path_chunks.extend(chunk_ids)
                            relevant_chunks.update(chunk_ids)

                # Add formatted source reference for this path
                if path_chunks:
                    source_ref = self._format_source_reference(path_chunks)
                    context_parts.append(f"  {source_ref}")

        # Add source text snippets
        if relevant_chunks and self.text_units:
            context_parts.append("\n## Source Text Snippets:")
            # Limit to top 10 most relevant chunks
            for idx, chunk_id in enumerate(list(relevant_chunks)[:10], 1):
                if chunk_id in self.text_units:
                    text = self.text_units[chunk_id]
                    # Truncate if too long
                    if len(text) > 500:
                        preview = text[:500].replace("\n", " ") + "..."
                    else:
                        preview = text.replace("\n", " ")
                    context_parts.append(f"\n**[{chunk_id}]**")
                    context_parts.append(preview)

            if len(relevant_chunks) > 10:
                context_parts.append(f"\n... ({len(relevant_chunks) - 10} more source chunks omitted)")

        context = "\n".join(context_parts)

        # Create prompt based on mode
        if strict_mode:
            system_prompt = """You are a knowledge graph reasoning assistant. 

CRITICAL CONSTRAINTS:
1. You MUST ONLY use information from the provided knowledge graph reasoning results and source text snippets
2. You MUST NOT use your own training data or general knowledge
3. If the provided reasoning results don't contain enough information to answer, say "Based on the available knowledge graph data, I cannot find sufficient information to answer this question."
4. Every statement in your answer must be traceable to the provided entities, reasoning paths, and source texts
5. Do not make assumptions or inferences beyond what is explicitly stated in the reasoning results
6. When making claims, reference the sources using the format to support your statements"""

            prompt = f"""Question: {query}

Knowledge Graph Reasoning Results:
{context}

IMPORTANT: Answer ONLY based on the above reasoning results and source texts. Do NOT use any external knowledge or your training data. If the reasoning results are insufficient, explicitly state that. When possible, cite the sources using the [source: ...] format shown above to support your answer."""

        else:
            system_prompt = "You are a helpful assistant that answers questions based on knowledge graph reasoning."

            prompt = f"""Based on the following knowledge graph reasoning results, please answer the question.

Question: {query}

{context}

Please provide a comprehensive answer based on the reasoning paths, entities, and source texts above. 
Explain how the paths support your answer and cite source chunks when relevant."""

        try:
            response = self.client.chat.completions.create(
                model=self.generation_model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature
            )

            answer = response.choices[0].message.content
            return answer

        except Exception as e:
            logging.error(f"Answer generation failed: {e}")
            return "Error generating answer. Please try again."

    def query(self, query_text: str, method: str = 'ppr',
              include_llm_answer: bool = True, strict_mode: bool = True) -> Dict[str, Any]:
        """
        Main query interface.

        Args:
            query_text: Natural language query
            method: Reasoning method ('ppr' or 'gnn')
            include_llm_answer: Whether to generate final answer with LLM
            strict_mode: If True, LLM only uses knowledge graph data (no external knowledge)

        Returns:
            Complete query results including reasoning and answer
        """
        logging.info("=" * 80)
        logging.info(f"Processing query: {query_text}")
        logging.info(f"Mode: method={method}, include_llm={include_llm_answer}, strict_mode={strict_mode}")
        logging.info("=" * 80)

        # 1. Encode query
        query_embedding_np = self.encode_query(query_text)
        query_embedding = torch.from_numpy(query_embedding_np).float().to(self.device)

        # --- FIX: Handle device mismatch similar to trainer.py ---
        # The GraphReasoner instance expects data on same device as model/query.
        # Since we keep graph_data on CPU for memory optimization, we must
        # temporarily move necessary tensors to the GPU for the reasoning pass.

        original_devices = {}
        tensors_to_move = [
            'node_embeddings', 'edge_index', 'edge_types',
            'edge_weights', 'edge_type_embeddings'
        ]

        try:
            # Move graph tensors to device (if not already there)
            if self.device.type == 'cuda':
                logging.debug(f"🔧 Temporarily moving graph tensors to {self.device} for reasoning...")
                for attr in tensors_to_move:
                    if hasattr(self.graph_data, attr):
                        tensor = getattr(self.graph_data, attr)
                        if isinstance(tensor, torch.Tensor):
                            original_devices[attr] = tensor.device
                            if tensor.device != self.device:
                                # In-place update of graph_data reference so reasoner sees it
                                setattr(self.graph_data, attr, tensor.to(self.device))

            # 2. Perform graph reasoning
            reasoning_results = self.reasoner.reason(
                query_text=query_text,
                text_encoder=self.encode_query,
                method=method
            )

        finally:
            # Restore to original device (likely CPU) to save memory
            if original_devices:
                logging.debug("🔧 Restoring graph tensors to original devices...")
                for attr, device in original_devices.items():
                    tensor = getattr(self.graph_data, attr)
                    if tensor.device != device:
                        setattr(self.graph_data, attr, tensor.to(device))

                # Clear GPU cache
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()

        # 3. Generate final answer (optional)
        if include_llm_answer:
            answer = self.generate_answer(query_text, reasoning_results, strict_mode=strict_mode)
            reasoning_results['answer'] = answer

        logging.info(f"Query processing complete")
        logging.info(f"  Top nodes: {len(reasoning_results['top_nodes'])}")
        logging.info(f"  Reasoning paths: {reasoning_results['num_paths']}")
        logging.info("=" * 80)

        return reasoning_results

    def train(self, num_epochs: Optional[int] = None):
        """
        Train the reasoning model.

        Args:
            num_epochs: Number of training epochs (uses config if None)
        """
        logging.info("Starting graph reasoning model training...")

        trainer = GraphReasoningTrainer(
            config=self.config,
            graph_data=self.graph_data,
            device=self.device
        )

        # Train
        history = trainer.train(num_epochs=num_epochs)

        # Save final model
        model_path = PROJECT_ROOT / self.reasoning_config.get('output', {}).get('model_path', 'data/reasoning/model.pt')
        model_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'epoch': trainer.epoch,
            'gnn_state_dict': trainer.gnn.state_dict(),
            'link_decoder_state_dict': trainer.link_decoder.state_dict(),
            'query_matcher_state_dict': trainer.query_matcher.state_dict(),
            'best_loss': trainer.best_loss,
            'config': self.config,
            'history': history
        }, model_path)

        logging.info(f"Training complete. Model saved to {model_path}")

        # Update local references
        self.gnn = trainer.gnn
        self.query_matcher = trainer.query_matcher

        # Recreate reasoner with trained models
        self.reasoner = GraphReasoner(
            config=self.config,
            graph_data=self.graph_data,
            gnn_model=self.gnn,
            query_matcher=self.query_matcher,
            device=self.device
        )

        return history


def print_results(results: Dict[str, Any], handler: Optional['ReasoningQueryHandler'] = None):
    """
    Format and print reasoning results with source references.

    Args:
        results: Reasoning results dictionary
        handler: Optional ReasoningQueryHandler instance for formatting source references
    """
    print("\n" + "=" * 80)
    print("REASONING RESULTS")
    print("=" * 80)
    print(f"\nQuery: {results['query']}")

    # Top entities
    print(f"\n{'Top Relevant Entities:':<30}")
    print("-" * 80)
    for i, node in enumerate(results['top_nodes'][:10], 1):
        # Display node with source reference if available
        chunk_ids = node.get('chunk_ids', [])
        if chunk_ids and handler and handler.chunk_id_map:
            source_ref = handler._format_source_reference(chunk_ids)
            print(f"{i:2}. {node['name']:<50} (score: {node['score']:.4f}) {source_ref}")
        else:
            print(f"{i:2}. {node['name']:<50} (score: {node['score']:.4f})")

    # Reasoning paths
    print(f"\n{'Reasoning Paths:':<30}")
    print("-" * 80)
    if results['paths']:
        for i, path_info in enumerate(results['paths'][:5], 1):
            print(f"\nPath {i} (confidence: {path_info['score']:.4f}):")
            print(path_info['explanation'])

            # Show source references for this path
            chunk_ids = path_info.get('chunk_ids', [])
            if chunk_ids and handler and handler.chunk_id_map:
                source_ref = handler._format_source_reference(chunk_ids)
                print(f"  {source_ref}")
    else:
        print("No reasoning paths found.")

    # Final answer
    if 'answer' in results:
        print(f"\n{'Final Answer:':<30}")
        print("-" * 80)
        print(results['answer'])

    print("\n" + "=" * 80)


def interactive_mode(config: Dict[str, Any]):
    """
    Interactive mode for reasoning queries.

    Args:
        config: Configuration dictionary
    """
    print("\n" + "=" * 80)
    print("Graph Reasoning Query System - Interactive Mode")
    print("=" * 80)

    # Check if model exists
    model_path = PROJECT_ROOT / config.get('reasoning', {}).get('output', {}).get('model_path', 'data/reasoning/model.pt')
    model_exists = model_path.exists()

    if model_exists:
        print(f"\n✓ Found trained model: {model_path}")
        load_model = True
    else:
        print(f"\n⚠ No trained model found at: {model_path}")
        train_choice = input("Do you want to train the model now? (yes/no) [yes]: ").strip().lower()

        if train_choice in ['', 'yes', 'y']:
            epochs_input = input("Number of training epochs [100]: ").strip()
            num_epochs = int(epochs_input) if epochs_input else 100

            print("\n" + "=" * 80)
            print("Starting Model Training...")
            print("=" * 80)

            # Train model
            handler = ReasoningQueryHandler(config, load_trained_model=False)
            handler.train(num_epochs=num_epochs)
            load_model = True

            print("\n✓ Training complete!")
        else:
            print("\n⚠ Cannot proceed without a trained model. Exiting.")
            return

    # Load handler
    print("\nLoading reasoning model...")
    handler = ReasoningQueryHandler(config, load_trained_model=load_model)
    print("✓ Model loaded successfully!\n")

    # Query loop
    while True:
        print("\n" + "-" * 80)
        query = input("\nEnter your query (or 'quit' to exit): ").strip()

        if query.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break

        if not query:
            print("⚠ Please enter a valid query.")
            continue

        # Choose method
        method_input = input("Reasoning method (ppr/gnn) [ppr]: ").strip().lower()
        method = method_input if method_input in ['ppr', 'gnn'] else 'ppr'

        # Choose whether to generate answer
        llm_input = input("Generate LLM answer? (yes/no) [yes]: ").strip().lower()
        include_llm = llm_input not in ['no', 'n']

        # Choose strict mode (only if generating LLM answer)
        strict_mode = True  # Default
        if include_llm:
            strict_input = input("Use strict mode (LLM only uses knowledge graph data)? (yes/no) [yes]: ").strip().lower()
            strict_mode = strict_input not in ['no', 'n']

        # Save to file?
        save_input = input("Save results to file? (yes/no) [no]: ").strip().lower()
        output_file = None
        if save_input in ['yes', 'y']:
            output_file = input("Output file path [data/reasoning/query_result.json]: ").strip()
            if not output_file:
                output_file = "data/reasoning/query_result.json"

        # Run query
        print("\n" + "=" * 80)
        print("Processing query...")
        print("=" * 80)

        try:
            results = handler.query(
                query_text=query,
                method=method,
                include_llm_answer=include_llm,
                strict_mode=strict_mode
            )

            # Print results
            print_results(results)

            # Write human-readable results to log
            log_lines = []
            log_lines.append("\n" + "=" * 80)
            log_lines.append("REASONING RESULTS")
            log_lines.append("=" * 80)
            log_lines.append(f"Query: {results.get('query', query)}")
            log_lines.append("\nTop Relevant Entities:")
            for i, node in enumerate(results.get('top_nodes', [])[:10], 1):
                log_lines.append(f"{i}. {node.get('name', '<unknown>')} (score: {node.get('score', 0):.4f})")

            log_lines.append("\nReasoning Paths:")
            paths = results.get('paths', [])
            if paths:
                for i, path_info in enumerate(paths[:5], 1):
                    log_lines.append(f"\nPath {i} (confidence: {path_info.get('score', 0):.4f}):")
                    log_lines.append(path_info.get('explanation', '').strip())
            else:
                log_lines.append("No reasoning paths found.")

            if 'answer' in results:
                log_lines.append("\nFinal Answer:")
                log_lines.append(results.get('answer', '').strip())

            logging.info("\n".join(log_lines))

            # Also log full JSON for traceability
            logging.info("Reasoning results (JSON):\n%s", json.dumps(results, ensure_ascii=False, indent=2))

            # Save if requested
            if output_file:
                output_path = PROJECT_ROOT / output_file
                output_path.parent.mkdir(parents=True, exist_ok=True)

                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)

                print(f"\n✓ Results saved to: {output_path}")

        except Exception as e:
            print(f"\n✗ Error processing query: {e}")
            logging.error(f"Query processing error: {e}", exc_info=True)


def command_line_mode():
    """
    Command-line mode with argument parsing.
    """
    import argparse

    # Load config first to get default values
    config = load_config()
    setup_logging(config)

    # Get training defaults from config
    training_config = config.get('reasoning', {}).get('training', {})
    default_epochs = training_config.get('num_epochs', 100)
    default_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser = argparse.ArgumentParser(
        description="Graph Reasoning Query System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Query mode
  python reasoning_query_qwen.py --query "What is nickel used for?"

  # Interactive mode
  python reasoning_query_qwen.py --interactive

  # Training mode (uses config defaults: epochs={default_epochs}, device={default_device})
  python reasoning_query_qwen.py --train

  # Training mode with custom parameters
  python reasoning_query_qwen.py --train --epochs 500 --device cpu
        """
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--interactive', '-i', action='store_true',
                            help='Run in interactive mode')
    mode_group.add_argument('--train', action='store_true',
                            help='Train the model (skip if model exists)')

    # Query parameters
    parser.add_argument('--query', '-q', type=str, default=None,
                        help='Natural language query')
    parser.add_argument('--method', '-m', type=str, default='ppr',
                        choices=['ppr', 'gnn'],
                        help='Reasoning method (default: ppr)')
    parser.add_argument('--no-llm', action='store_true',
                        help='Skip LLM answer generation')
    parser.add_argument('--no-strict', action='store_true',
                        help='Allow LLM to use its own knowledge (default: strict mode - only knowledge graph data)')

    # Training parameters (defaults from config)
    parser.add_argument('--epochs', '-e', type=int, default=default_epochs,
                        help=f'Number of training epochs (default from config: {default_epochs})')
    parser.add_argument('--device', '-d', type=str, default=default_device,
                        choices=['cuda', 'cpu'],
                        help=f'Training device (default: {default_device})')
    parser.add_argument('--batch-size', '-b', type=int, default=None,
                        help=f'Training batch size (default from config: {training_config.get("batch_size", 256)}). Reduce if out of memory.')
    parser.add_argument('--gradient-accumulation', '-g', type=int, default=None,
                        help=f'Gradient accumulation steps (default: {training_config.get("gradient_accumulation_steps", 2)}). Increase if out of memory.')
    parser.add_argument('--force-train', action='store_true',
                        help='Force training even if model exists')

    # Output
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Save results to JSON file')

    args = parser.parse_args()

    # Check model existence
    model_path = PROJECT_ROOT / config.get('reasoning', {}).get('output', {}).get('model_path', 'data/reasoning/model.pt')
    model_exists = model_path.exists()

    # Interactive mode
    if args.interactive:
        interactive_mode(config)
        return

    # Training mode
    if args.train or args.force_train:
        if model_exists and not args.force_train:
            print(f"\n✓ Model already exists at: {model_path}")
            print("  Use --force-train to retrain anyway.")
            print("  Skipping training.\n")
        else:
            print("\n" + "=" * 80)
            print("Starting Model Training...")
            print("=" * 80)

            # Determine batch size
            batch_size = args.batch_size if args.batch_size else training_config.get('batch_size', 256)

            # Determine gradient accumulation (default to 2 for memory safety)
            gradient_accumulation = args.gradient_accumulation if args.gradient_accumulation is not None else training_config.get('gradient_accumulation_steps', 2)

            # Memory optimization tips
            if args.device == 'cuda':
                print(f"GPU Memory Optimization Tips:")
                print(f"  - If you encounter OOM (Out of Memory) errors:")
                print(f"    1. Reduce batch size: --batch-size 128 or --batch-size 64")
                print(f"    2. Use gradient accumulation: --gradient-accumulation 4 or 8")
                print(f"    3. Try CPU mode: --device cpu (slower but no memory limit)")

            print(f"\nTraining Configuration:")
            print(f"  - Epochs: {args.epochs}")
            print(f"  - Device: {args.device}")
            print(f"  - Batch Size: {batch_size}")
            print(f"  - Gradient Accumulation Steps: {gradient_accumulation}")
            print(f"  - Learning Rate: {training_config.get('learning_rate', 0.001)}")
            print(f"  - Weight Decay: {training_config.get('weight_decay', 0.0001)}")
            print("=" * 80)

            # Override batch size in config if specified
            if args.batch_size:
                config['reasoning']['training']['batch_size'] = args.batch_size

            # Add gradient accumulation to config
            config['reasoning']['training']['gradient_accumulation_steps'] = gradient_accumulation

            handler = ReasoningQueryHandler(config, load_trained_model=False)

            # CRITICAL: Clear GPU cache before training
            if args.device == 'cuda':
                torch.cuda.empty_cache()
                logging.info("🔧 Cleared GPU cache before training")

            # Override device if specified
            if args.device:
                handler.device = args.device
                logging.info(f"🔧 Setting training device to: {args.device}")

                # Move ONLY models to device, NOT graph data
                handler.gnn = handler.gnn.to(args.device)
                handler.query_matcher = handler.query_matcher.to(args.device)

                # CRITICAL CHANGE: Keep ALL graph data on CPU
                # The trainer will handle moving data to GPU during forward pass
                logging.info("🔧 [Memory Optimization] ALL graph data kept on CPU (trainer will handle device transfer)")
                logging.info(f"  - node_embeddings: {handler.graph_data.node_embeddings.device}")
                logging.info(f"  - edge_index: {handler.graph_data.edge_index.device}")
                logging.info(f"  - edge_types: {handler.graph_data.edge_types.device}")
                logging.info(f"  - edge_weights: {handler.graph_data.edge_weights.device}")

                # Clear cache after model initialization
                if args.device == 'cuda':
                    torch.cuda.empty_cache()
                    allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
                    reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
                    logging.info(f"📊 GPU Memory after setup: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

            handler.train(num_epochs=args.epochs)

            print("\n✓ Training complete!")
        return

    # Query mode
    if args.query:
        print("\n" + "=" * 80)
        print("Graph Reasoning Query System")
        print("=" * 80)

        if not model_exists:
            print(f"\n✗ Error: No trained model found at {model_path}")
            print("  Please train the model first using: --train")
            return

        # Load handler
        print("\nLoading reasoning model...")
        handler = ReasoningQueryHandler(config, load_trained_model=True)

        # Run query
        print(f"\nProcessing query: {args.query}")
        results = handler.query(
            query_text=args.query,
            method=args.method,
            include_llm_answer=not args.no_llm,
            strict_mode=not args.no_strict
        )

        # Print results
        print_results(results, handler=handler)

        # Save if requested
        if args.output:
            output_path = PROJECT_ROOT / args.output
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            print(f"\n✓ Results saved to: {output_path}")
    else:
        # No arguments - show help or enter interactive
        print("\nNo query provided. Starting interactive mode...\n")
        interactive_mode(config)


def main():
    """Main entry point"""
    try:
        command_line_mode()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        logging.error(f"Fatal error: {e}", exc_info=True)


if __name__ == "__main__":
    main()