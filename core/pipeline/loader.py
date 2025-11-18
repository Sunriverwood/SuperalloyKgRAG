# core/pipeline/loader.py
import hashlib
import json
import os
from typing import Any, Dict, List, Literal
import pandas as pd
from pydantic import BaseModel, Field

# --- 配置参数 ---
# 源数据文件夹
SOURCE_JSON_DIR = "../../data/processed_jsons/"
# 分块后数据存储文件夹
OUTPUT_DIR = "../../data/chunks/"
# 分块后文件名
FILENAME_MAP = {
    "jsonl": "text_units.jsonl",
    "json": "text_units.json",
    "parquet": "text_units.parquet",
}
OUTPUT_FORMAT: Literal['jsonl', 'json', 'parquet'] = 'jsonl'

# GraphRAG 风格的分块策略参数
CHUNK_SIZE = 500  # 每个文本块的目标大小 (字符数)
CHUNK_OVERLAP = 100  # 相邻文本块之间的重叠大小 (字符数)


# =================================================================
# 1. 引入结构化数据模型 (参考 GraphRAG 的 data_model)
# =================================================================
class TextSegment(BaseModel):
    """
    代表一个带有位置信息的文本片段（来自特定的page和block）。
    """
    text: str = Field(description="文本片段的内容。")
    page_number: Any = Field(description="文本片段所在的页码。")
    block_id: Any = Field(description="文本片段所在的block ID。")
    start_pos: int = Field(description="在完整文档文本中的起始位置。")
    end_pos: int = Field(description="在完整文档文本中的结束位置。")

class Document(BaseModel):
    """
    代表一个完整的、从源文件加载的文档。
    """
    id: str = Field(description="文档的唯一ID，通常是源文件名的哈希值。")
    text: str = Field(description="从文档中提取的全部纯文本内容。")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="文档的元数据，如源文件名、页码等。")
    text_segments: List[TextSegment] = Field(default_factory=list, description="文档中的文本片段及其位置信息。")

class TextUnit(BaseModel):
    """
    代表一个文本块（Chunk），是图谱提取的基本单元。
    """
    id: str = Field(description="文本单元的唯一ID。")
    document_id: str = Field(description="来源文档的ID，用于数据溯源。")
    text: str = Field(description="文本块的具体内容。")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="文本单元的元数据，如在文档中的起止位置等。")

# =================================================================
# 2. 封装核心逻辑到类中，提升模块化
# =================================================================
class DocumentLoader:
    """
    一个负责加载、处理和分块文档的类，使其成为一个独立的流水线阶段。
    """
    def __init__(self, source_dir: str, output_dir: str, chunk_size: int, chunk_overlap: int):
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self, output_format: str = OUTPUT_FORMAT) -> str:
        """
        执行加载和分块的完整流程。
        """
        print("Starting the document loading and chunking process...")
        # 步骤1: 从源文件夹加载所有文档
        documents = self._load_all_documents()
        if not documents:
            print(f"No processable JSON files found in '{self.source_dir}'.")
            return ""

        print(f"Successfully loaded {len(documents)} documents.")

        # 步骤2: 对所有文档进行分块，生成文本单元
        text_units = self._chunk_all_documents(documents)
        print(f"Generated {len(text_units)} text units.")

        # 步骤3: 将文本单元保存为统一的结构化文件
        output_path = self._save_text_units(text_units, output_format)
        print(f"All text units have been saved to '{output_path}'.")

        print("\nProcess finished successfully!")
        return output_path

    def _load_all_documents(self) -> List[Document]:
        """
        遍历源目录，加载所有JSON文件为Document对象。
        """
        documents = []
        json_files = [f for f in os.listdir(self.source_dir) if f.endswith(".json")]

        for json_file in json_files:
            file_path = os.path.join(self.source_dir, json_file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                document = self._extract_text_from_json(data, json_file)
                if document.text.strip():  # 仅添加有实质内容的文档
                    documents.append(document)
                else:
                    print(f"  -> Warning: No text content extracted from {json_file}. Skipping.")

            except json.JSONDecodeError:
                print(f"  -> Error: Could not decode JSON from {json_file}.")
            except Exception as e:
                print(f"  -> An unexpected error occurred while processing {json_file}: {e}")

        return documents

    def _extract_text_from_json(self, data: List[Dict[str, Any]], source_filename: str) -> Document:
        """
        从多个JSON页面中提取文本，并创建一个Document实例。
        """
        full_text_blocks = []
        text_segments = []
        current_position = 0

        # 如果输入是字典，转换为列表进行处理
        if isinstance(data, dict):
            data = [data]

        for page in data:
            page_number = page.get("page_number", "N/A")

            # 遍历每个页面的内容块
            for block in page.get("content_blocks", []):
                if block.get("type") == "text_block":
                    text_content = block.get("content", "").strip()
                    if text_content:
                        block_id = block.get('block_id', 'N/A')
                        # 添加源信息，但作为文本的一部分，以便LLM理解上下文
                        # metadata可以存储更结构化的信息
                        source_prefix = f"[Source: Page {page_number}, Block: {block_id}]"
                        full_block_text = f"{source_prefix}\n{text_content}"
                        full_text_blocks.append(full_block_text)

                        # 记录文本段的位置信息
                        segment_start = current_position
                        segment_end = current_position + len(full_block_text)
                        text_segments.append(TextSegment(
                            text=full_block_text,
                            page_number=page_number,
                            block_id=block_id,
                            start_pos=segment_start,
                            end_pos=segment_end
                        ))

                        # 更新位置（加上分隔符的长度）
                        current_position = segment_end + 2  # "\n\n" 的长度

        full_text = "\n\n".join(full_text_blocks)

        # 使用文件名哈希作为文档的稳定ID
        doc_id = hashlib.md5(source_filename.encode('utf-8')).hexdigest()

        return Document(
            id=f"doc-{doc_id}",
            text=full_text,
            metadata={"source_filename": source_filename},
            text_segments=text_segments
        )

    def _chunk_all_documents(self, documents: List[Document]) -> List[TextUnit]:
        """
        将所有Document对象分块成TextUnit对象列表。
        """
        all_text_units = []
        for doc in documents:
            chunks_with_metadata = self._chunk_text(doc.text, doc.text_segments)
            for i, (chunk_text, chunk_meta) in enumerate(chunks_with_metadata):
                chunk_id_hash = hashlib.md5(f"{doc.id}-{i}-{chunk_text}".encode('utf-8')).hexdigest()

                # 合并元数据
                metadata = {
                    "chunk_index": i,
                    "source_filename": doc.metadata["source_filename"],
                    "pages": chunk_meta["pages"],
                    "blocks": chunk_meta["blocks"]
                }

                text_unit = TextUnit(
                    id=f"chunk-{chunk_id_hash}",
                    document_id=doc.id,
                    text=chunk_text,
                    metadata=metadata
                )
                all_text_units.append(text_unit)
        return all_text_units

    def _chunk_text(self, text: str, text_segments: List[TextSegment]) -> List[tuple]:
        """
        将单篇长文本按固定大小和重叠进行分块，并返回每个chunk及其元数据。
        返回: List[Tuple[str, Dict]] - 每个元素包含(chunk_text, chunk_metadata)
        """
        if not text:
            return []

        chunks_with_metadata = []
        start_index = 0

        while start_index < len(text):
            end_index = start_index + self.chunk_size
            chunk_text = text[start_index:end_index]

            # 确定这个chunk跨越了哪些页面和block
            pages = set()
            blocks = set()

            for segment in text_segments:
                # 检查chunk是否与这个segment有重叠
                # chunk范围: [start_index, end_index)
                # segment范围: [segment.start_pos, segment.end_pos)
                if not (end_index <= segment.start_pos or start_index >= segment.end_pos):
                    # 有重叠
                    pages.add(segment.page_number)
                    blocks.add(segment.block_id)

            # 转换为排序后的列表，便于查看
            chunk_metadata = {
                "pages": sorted(list(pages), key=lambda x: (isinstance(x, str), x)),
                "blocks": sorted(list(blocks), key=lambda x: (isinstance(x, str), x))
            }

            chunks_with_metadata.append((chunk_text, chunk_metadata))

            # 如果是最后一块，则停止
            if end_index >= len(text):
                break
            start_index += self.chunk_size - self.chunk_overlap

        return chunks_with_metadata

    def _save_text_units(self, text_units: List[TextUnit], format: str) -> str:
        """
        根据指定的格式保存TextUnit列表。
        """
        output_path = os.path.join(self.output_dir, FILENAME_MAP.get(format, "text_units.jsonl"))

        if format == 'parquet':
            self._save_as_parquet(text_units, output_path)
        elif format == 'json':
            self._save_as_json(text_units, output_path)
        else:  # 默认或指定为 'jsonl'
            self._save_as_jsonl(text_units, output_path)

        return output_path

    def _save_as_parquet(self, text_units: List[TextUnit], path: str):
        """保存为 Parquet 文件。"""
        print(f"  -> Saving to Parquet format at {path}...")
        data_to_save = [unit.model_dump() for unit in text_units]
        df = pd.DataFrame(data_to_save)
        df = df[['id', 'document_id', 'text', 'metadata']]
        df.to_parquet(path, index=False)

    def _save_as_json(self, text_units: List[TextUnit], path: str):
        """保存为单个 JSON 文件。"""
        print(f"  -> Saving to JSON format at {path}...")
        data_to_save = [unit.model_dump() for unit in text_units]
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=4)

    def _save_as_jsonl(self, text_units: List[TextUnit], path: str):
        """保存为 JSON Lines 文件。"""
        print(f"  -> Saving to JSONL format at {path}...")
        with open(path, 'w', encoding='utf-8') as f:
            for unit in text_units:
                # 将每个Pydantic模型转换为JSON字符串并写入一行
                f.write(unit.model_dump_json() + '\n')

# =================================================================
# 3. 主执行入口
# =================================================================
if __name__ == "__main__":
    # 初始化并运行加载器
    loader = DocumentLoader(
        source_dir=SOURCE_JSON_DIR,
        output_dir=OUTPUT_DIR,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    loader.run()