"""
完整索引流水线执行脚本
实现从PDF文档到知识图谱向量存储的端到端数据处理流程

流程步骤：
1. OCR解析 (vlm_pdf_parser) - 将PDF转换为结构化JSON
2. 文本分块 (loader) - 将JSON文档切分为文本单元
3. 三元组提取 (extraction) - 从文本块中提取实体和关系
4. 图谱构建 (graph_builder) - 消歧、合并、社区发现
5. 向量化存储 (embedding) - 将图谱数据嵌入向量数据库
"""

import sys
import logging
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
import yaml

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# 导入各流水线模块
from core.vlm_pdf_parser_qwen import main as vlm_parser_main
from core.pipeline_qwen.loader import DocumentLoader
from core.pipeline_qwen.extraction_qwen import run_extraction
from core.pipeline_qwen.graph_builder_qwen import main as graph_builder_main
from core.pipeline_qwen.embedding_qwen import main as embedding_main

# =========================
# 配置与日志
# =========================

def load_config(settings_filename: str = "settings.yaml") -> Dict[str, Any]:
    """加载YAML配置文件"""
    config_path = PROJECT_ROOT / "config" / settings_filename
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件 {config_path} 未找到！")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(config: Dict[str, Any]):
    """根据配置文件设置日志记录器"""
    log_config = config.get("logging", {})
    level = getattr(logging, log_config.get("level", "INFO").upper(), logging.INFO)
    relative_log_path = log_config.get("log_file", "logs/run_indexing_qwen.log")
    log_file = PROJECT_ROOT / relative_log_path

    log_file.parent.mkdir(exist_ok=True, parents=True)

    # 移除所有现有的处理器
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='a', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logging.info("=" * 80)
    logging.info("索引流水线日志记录器设置完成")
    logging.info("=" * 80)


# =========================
# 流程状态管理
# =========================

class PipelineStateManager:
    """管理流水线执行状态，支持断点续传"""

    STEPS = {
        1: "ocr_parsing",
        2: "text_chunking",
        3: "triple_extraction",
        4: "graph_building",
        5: "vector_embedding"
    }

    def __init__(self, state_file: Path):
        self.state_file = state_file
        self.state_file.parent.mkdir(exist_ok=True, parents=True)
        self.state = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        """加载流水线执行状态"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                logging.info(f"加载流水线状态: {self.state_file}")
                return state
            except json.JSONDecodeError:
                logging.warning(f"状态文件损坏，创建新状态: {self.state_file}")
                return self._create_new_state()
        else:
            return self._create_new_state()

    def _create_new_state(self) -> Dict[str, Any]:
        """创建新的状态记录"""
        return {
            "last_completed_step": 0,
            "last_run_time": None,
            "steps": {name: {"completed": False, "timestamp": None, "duration": None}
                     for name in self.STEPS.values()}
        }

    def _save_state(self):
        """保存状态到文件"""
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(self.state, f, indent=2, ensure_ascii=False)

    def mark_step_start(self, step_num: int):
        """标记步骤开始"""
        step_name = self.STEPS[step_num]
        self.state["steps"][step_name]["start_time"] = datetime.now().isoformat()
        self._save_state()

    def mark_step_complete(self, step_num: int, duration: float):
        """标记步骤完成"""
        step_name = self.STEPS[step_num]
        self.state["steps"][step_name]["completed"] = True
        self.state["steps"][step_name]["timestamp"] = datetime.now().isoformat()
        self.state["steps"][step_name]["duration"] = f"{duration:.2f}s"
        self.state["last_completed_step"] = step_num
        self.state["last_run_time"] = datetime.now().isoformat()
        self._save_state()
        logging.info(f"✅ 步骤 {step_num} ({step_name}) 已完成，耗时: {duration:.2f}s")

    def get_last_completed_step(self) -> int:
        """获取最后完成的步骤"""
        return self.state.get("last_completed_step", 0)

    def reset(self):
        """重置状态"""
        self.state = self._create_new_state()
        self._save_state()
        logging.info("流水线状态已重置")


# =========================
# 步骤依赖验证
# =========================

class DependencyValidator:
    """验证各步骤的输入文件依赖"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.project_root = PROJECT_ROOT

    def validate_step_1(self) -> bool:
        """验证步骤1：OCR解析 - 检查PDF文件是否存在"""
        input_dir = self.project_root / self.config["vlm_parser"]["input_dir"]
        if not input_dir.exists():
            logging.error(f"�� 步骤1依赖验证失败: 输入目录不存在 {input_dir}")
            return False

        pdf_files = list(input_dir.glob("*.pdf"))
        if not pdf_files:
            logging.error(f"❌ 步骤1依赖验证失败: 在 {input_dir} 中未找到PDF文件")
            return False

        logging.info(f"✅ 步骤1依赖验证通过: 发现 {len(pdf_files)} 个PDF文件")
        return True

    def validate_step_2(self) -> bool:
        """验证步骤2：文本分块 - 检查JSON文件是否存在"""
        json_dir = self.project_root / self.config["vlm_parser"]["output_dir"]
        if not json_dir.exists():
            logging.error(f"❌ 步骤2依赖验证失败: JSON目录不存在 {json_dir}")
            return False

        json_files = list(json_dir.glob("*.json"))
        if not json_files:
            logging.error(f"❌ 步骤2依赖验证失败: 在 {json_dir} 中未找到JSON文件")
            logging.info("💡 提示: 请先运行步骤1 (OCR解析)")
            return False

        logging.info(f"✅ 步骤2依赖验证通过: 发现 {len(json_files)} 个JSON文件")
        return True

    def validate_step_3(self) -> bool:
        """验证步骤3：三元组提取 - 检查文本单元文件是否存在"""
        chunks_file = self.project_root / self.config["loader"]["output_dir"] / self.config["loader"]["filename_map"]["jsonl"]
        if not chunks_file.exists():
            logging.error(f"❌ 步骤3依赖验证失败: 文本单元文件不存在 {chunks_file}")
            logging.info("💡 提示: 请先运行步骤2 (文本分块)")
            return False

        # 检查文件是否为空
        with open(chunks_file, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            if not first_line.strip():
                logging.error(f"❌ 步骤3依赖验证失败: 文本单元文件为空 {chunks_file}")
                return False

        logging.info(f"✅ 步骤3依赖验证通过: 文本单元文件存在且非空")
        return True

    def validate_step_4(self) -> bool:
        """验证步骤4：图谱构建 - 检查提取的图谱文件是否存在"""
        extracted_graph = self.project_root / self.config["extraction"]["output_dir"] / self.config["extraction"]["output_filename"]
        if not extracted_graph.exists():
            logging.error(f"❌ 步骤4依赖验证失败: 提取的图谱文件不存在 {extracted_graph}")
            logging.info("💡 提示: 请先运行步骤3 (三元组提取)")
            return False

        # 检查文件是否为空
        with open(extracted_graph, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            if not first_line.strip():
                logging.error(f"❌ 步骤4依赖验证失败: 提取的图谱文件为空 {extracted_graph}")
                return False

        logging.info(f"✅ 步骤4依赖验证通过: 提取的图谱文件存在且非空")
        return True

    def validate_step_5(self) -> bool:
        """验证步骤5：向量化存储 - 检查最终图谱和社区报告是否存在"""
        final_graph = self.project_root / self.config["graph_builder"]["output_graph_path"]
        community_reports = self.project_root / self.config["graph_builder"]["community_reports_path"]

        if not final_graph.exists():
            logging.error(f"❌ 步骤5依赖验证失败: 最终图谱文件不存在 {final_graph}")
            logging.info("💡 提示: 请先运行步骤4 (图谱构建)")
            return False

        if not community_reports.exists():
            logging.error(f"❌ 步骤5依赖验证失败: 社区报告文件不存在 {community_reports}")
            logging.info("💡 提示: 请先运行步骤4 (图谱构建)")
            return False

        logging.info(f"✅ 步骤5依赖验证通过: 最终图谱和社区报告文件存在")
        return True

    def validate(self, step_num: int) -> bool:
        """验证指定步骤的依赖"""
        validators = {
            1: self.validate_step_1,
            2: self.validate_step_2,
            3: self.validate_step_3,
            4: self.validate_step_4,
            5: self.validate_step_5
        }

        validator = validators.get(step_num)
        if validator:
            return validator()
        else:
            logging.warning(f"⚠️ 未找到步骤 {step_num} 的验证器")
            return True


# =========================
# 流水线执行器
# =========================

class IndexingPipeline:
    """索引流水线执行器"""

    def __init__(self, config: Dict[str, Any], state_manager: PipelineStateManager,
                 validator: DependencyValidator):
        self.config = config
        self.state_manager = state_manager
        self.validator = validator

    def run_step_1_ocr_parsing(self):
        """步骤1: OCR解析"""
        logging.info("\n" + "=" * 80)
        logging.info("🚀 开始执行步骤1: OCR解析 (PDF → JSON)")
        logging.info("=" * 80)

        vlm_parser_main()

        logging.info("✅ 步骤1完成: OCR解析")

    def run_step_2_text_chunking(self):
        """步骤2: 文本分块"""
        logging.info("\n" + "=" * 80)
        logging.info("🚀 开始执行步骤2: 文本分块 (JSON → Text Units)")
        logging.info("=" * 80)

        loader_config = self.config["loader"]
        loader = DocumentLoader(
            source_dir=str(PROJECT_ROOT / loader_config["source_json_dir"]),
            output_dir=str(PROJECT_ROOT / loader_config["output_dir"]),
            chunk_size=loader_config["chunk_size"],
            chunk_overlap=loader_config["chunk_overlap"]
        )
        output_path = loader.run(output_format=loader_config["output_format"])

        logging.info(f"✅ 步骤2完成: 文本分块，输出文件: {output_path}")

    def run_step_3_triple_extraction(self):
        """步骤3: 三元组提取"""
        logging.info("\n" + "=" * 80)
        logging.info("🚀 开始执行步骤3: 三元组提取 (Text Units → Graph Triples)")
        logging.info("=" * 80)

        run_extraction()

        logging.info("✅ 步骤3完成: 三元组提取")

    def run_step_4_graph_building(self):
        """步骤4: 图谱构建"""
        logging.info("\n" + "=" * 80)
        logging.info("🚀 开始执行步骤4: 图谱构建 (消歧 → 合并 → 社区发现)")
        logging.info("=" * 80)

        graph_builder_main()

        logging.info("✅ 步骤4完成: 图谱构建")

    def run_step_5_vector_embedding(self):
        """步骤5: 向量化存储"""
        logging.info("\n" + "=" * 80)
        logging.info("🚀 开始执行步骤5: 向量化存储 (Graph → Vector DB)")
        logging.info("=" * 80)

        embedding_main()

        logging.info("✅ 步骤5完成: 向量化存储")

    def execute_step(self, step_num: int) -> bool:
        """执行单个步骤"""
        step_functions = {
            1: self.run_step_1_ocr_parsing,
            2: self.run_step_2_text_chunking,
            3: self.run_step_3_triple_extraction,
            4: self.run_step_4_graph_building,
            5: self.run_step_5_vector_embedding
        }

        step_func = step_functions.get(step_num)
        if not step_func:
            logging.error(f"❌ 未找到步骤 {step_num} 的执行函数")
            return False

        # 依赖验证
        if not self.validator.validate(step_num):
            logging.error(f"❌ 步骤 {step_num} 依赖验证失败，跳过执行")
            return False

        # 执行步骤
        try:
            self.state_manager.mark_step_start(step_num)
            start_time = time.time()

            step_func()

            duration = time.time() - start_time
            self.state_manager.mark_step_complete(step_num, duration)
            return True

        except Exception as e:
            logging.error(f"❌ 步骤 {step_num} 执行失败: {e}", exc_info=True)
            return False

    def run_full_pipeline(self, start_from: int = 1, end_at: int = 5):
        """执行完整流水线"""
        logging.info("\n" + "=" * 80)
        logging.info(f"🎯 开始执行完整索引流水线 (步骤 {start_from} → {end_at})")
        logging.info("=" * 80)

        total_start_time = time.time()

        for step_num in range(start_from, end_at + 1):
            success = self.execute_step(step_num)
            if not success:
                logging.error(f"❌ 流水线在步骤 {step_num} 中断")
                return False

        total_duration = time.time() - total_start_time
        logging.info("\n" + "=" * 80)
        logging.info(f"🎉 完整索引流水线执行完成！总耗时: {total_duration:.2f}s")
        logging.info("=" * 80)
        return True


# =========================
# 主程序入口
# =========================

def main():
    """主执行函数"""
    parser = argparse.ArgumentParser(
        description="SuperalloyKgRAG 索引流水线执行脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
执行示例:
  python run_indexing_qwen.py                    # 执行完整流水线 (步骤1-5)
  python run_indexing_qwen.py --start 3          # 从步骤3开始执行到步骤5
  python run_indexing_qwen.py --start 2 --end 4  # 执行步骤2-4
  python run_indexing_qwen.py --step 5           # 仅执行步骤5
  python run_indexing_qwen.py --reset            # 重置流水线状态后执行
  python run_indexing_qwen.py --resume           # 从上次中断处继续

步骤说明:
  1 - OCR解析 (PDF → JSON)
  2 - 文本分块 (JSON → Text Units)
  3 - 三元组提取 (Text Units → Graph Triples)
  4 - 图谱构建 (消歧 → 合并 → 社区发现)
  5 - 向量化存储 (Graph → Vector DB)
        """
    )

    parser.add_argument('--start', type=int, choices=[1, 2, 3, 4, 5],
                        help='起始步骤 (默认: 1)')
    parser.add_argument('--end', type=int, choices=[1, 2, 3, 4, 5],
                        help='结束步骤 (默认: 5)')
    parser.add_argument('--step', type=int, choices=[1, 2, 3, 4, 5],
                        help='仅执行指定步骤')
    parser.add_argument('--reset', action='store_true',
                        help='重置流水线状态后执行')
    parser.add_argument('--resume', action='store_true',
                        help='从上次中断处继续执行')

    args = parser.parse_args()

    try:
        # 加载配置
        config = load_config()
        setup_logging(config)

        # 初始化状态管理器
        state_file = PROJECT_ROOT / "data" / "cache" / "pipeline_state.json"
        state_manager = PipelineStateManager(state_file)

        # 初始化依赖验证器
        validator = DependencyValidator(config)

        # 初始化流水线执行器
        pipeline = IndexingPipeline(config, state_manager, validator)

        # 处理重置选项
        if args.reset:
            logging.info("🔄 重置流水线状态...")
            state_manager.reset()

        # 确定执行范围
        if args.step:
            # 仅执行单个步骤
            start_step = args.step
            end_step = args.step
        elif args.resume:
            # 从上次中断处继续
            last_completed = state_manager.get_last_completed_step()
            start_step = last_completed + 1
            end_step = 5
            if start_step > 5:
                logging.info("✅ 所有步骤已完成，无需继续执行")
                return
            logging.info(f"📍 从步骤 {start_step} 继续执行 (上次完成: 步骤 {last_completed})")
        else:
            # 执行指定范围
            start_step = args.start if args.start else 1
            end_step = args.end if args.end else 5

            # 验证范围
            if start_step > end_step:
                logging.error("❌ 起始步骤不能大于结束步骤")
                return

        # 执行流水线
        success = pipeline.run_full_pipeline(start_from=start_step, end_at=end_step)

        if success:
            logging.info("\n✅ 流水线执行成功完成！")
            sys.exit(0)
        else:
            logging.error("\n❌ 流水线执行失败")
            sys.exit(1)

    except FileNotFoundError as e:
        logging.critical(f"关键文件未找到: {e}")
        sys.exit(1)
    except Exception as e:
        logging.critical(f"程序执行时发生致命错误: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

