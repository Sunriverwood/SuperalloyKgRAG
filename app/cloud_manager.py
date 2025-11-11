import os
from google import genai
from google.api_core import exceptions
from datetime import datetime, timezone

# ================================
# 配置区
# ================================
# 配置代理 (如果需要)
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"


def select_and_delete_files(client, files_to_manage):
    """
    一个通用的函数，用于显示文件列表并处理用户的删除选择。
    """
    if not files_to_manage:
        print("✅ 列表中没有可管理的文件。")
        return

    print("\n" + "-" * 50)
    print("请输入要删除的文件序号（如 1,3,5 或 2-4,7），或输入 'all' 删除全部，或直接回车取消：")
    user_input = input("您的选择: ").strip().lower()

    if not user_input:
        print("\n🚫 操作已取消。")
        return
    if user_input == 'all':
        indices = list(range(1, len(files_to_manage) + 1))
    else:
        indices = set()
        try:
            for part in user_input.split(','):
                part = part.strip()
                if '-' in part:
                    start, end = part.split('-', 1)
                    if start.strip().isdigit() and end.strip().isdigit():
                        s, e = int(start), int(end)
                        if 1 <= s <= e <= len(files_to_manage):
                            indices.update(range(s, e + 1))
                elif part.isdigit():
                    idx = int(part)
                    if 1 <= idx <= len(files_to_manage):
                        indices.add(idx)
        except Exception:
            print("❌ 输入格式有误，操作取消。")
            return
        if not indices:
            print("❌ 未选择任何有效文件，操作取消。")
            return
        indices = sorted(indices)

    print("\n🔥 正在删除所选文件，请稍候...")
    for idx in indices:
        f = files_to_manage[idx - 1]
        display_name = f.display_name or "未知"
        try:
            client.files.delete(name=f.name)
            print(f"  - 已删除 {display_name} ({f.name})")
        except Exception as e:
            print(f"  - 🔥 删除 {display_name} 失败: {e}")
    print("\n✅ 文件删除操作完成！")


# ================================
# 1. 用户上传文件管理
# ================================
def manage_uploaded_files(client):
    """
    列出所有用户直接上传的文件（不含作业生成的文件），并允许删除。
    """
    print("\n" + "=" * 50)
    print("📁 用户上传文件管理")
    print("=" * 50)
    print("🔍 正在获取您已上传的文件列表...")

    try:
        # 启发式过滤：作业文件通常在 'corpora' 或 'batches' 路径下，以此区分
        uploaded_files = [
            f for f in client.files.list()
            if 'corpora' not in f.name and 'batches' not in f.name
        ]

        if not uploaded_files:
            print("✅ 未找到任何由您直接上传的文件。")
            return

        print(f"\n📄 找到了 {len(uploaded_files)} 个已上传的文件：")
        for idx, f in enumerate(uploaded_files, 1):
            display_name = f.display_name or "未知"
            print(f"  {idx}. 显示名称: {display_name:<40} 文件 ID: {f.name}")

        select_and_delete_files(client, uploaded_files)

    except Exception as e:
        print(f"🔥 获取或删除文件时发生错误: {e}")


# ================================
# 2. 作业结果文件管理 (优化版)
# ================================
def manage_job_result_files(client):
    """
    精确查找并管理由批处理作业成功后生成的结果文件。
    """
    print("\n" + "=" * 50)
    print("📊 作业结果文件管理")
    print("=" * 50)
    print("🔍 正在查找所有已成功作业的结果文件...")

    try:
        all_jobs = list(client.batches.list())
        deletable_files_map = {}

        print("\n--- 查找结果 ---")
        job_count = 0
        file_idx = 1
        for job in sorted(all_jobs, key=lambda j: j.create_time, reverse=True):
            if job.state == genai.types.JobState.JOB_STATE_SUCCEEDED and hasattr(job, 'results') and hasattr(job.results,
                                                                                             'output_file') and job.results.output_file:
                job_count += 1
                file_name = job.results.output_file.name
                display_name = job.display_name or "无名作业"

                try:
                    file_obj = client.files.get(name=file_name)
                    print(f"  {file_idx}. 作业 '{display_name}' (状态: {job.state.name})")
                    print(f"     ➔ 结果文件: {file_name}")
                    deletable_files_map[file_idx] = file_obj
                    file_idx += 1
                except exceptions.NotFound:
                    print(f"  - 作业 '{display_name}'")
                    print(f"    - 结果文件 '{file_name}' 未找到，可能已被删除。")
                except Exception as e:
                    print(f"  - 作业 '{display_name}'")
                    print(f"    - 🔥 获取文件 '{file_name}' 失败: {e}")

        if not deletable_files_map:
            print("\n✅ 没有找到任何可管理的作业结果文件。")
            return

        files_to_manage = [deletable_files_map[i] for i in sorted(deletable_files_map.keys())]
        select_and_delete_files(client, files_to_manage)

    except Exception as e:
        print(f"🔥 获取或管理作业结果时发生错误: {e}")


# ================================
# 3. 批处理作业管理 (优化版)
# ================================
def manage_batch_jobs(client):
    """
    列出并管理正在运行的批处理作业，可以终止运行超时的作业。
    """
    print("\n" + "=" * 50)
    print("⚙️  批处理作业管理")
    print("=" * 50)

    try:
        while True:
            threshold_input = input("请输入超时阈值（小时），超过此时长的正在运行的作业将被列出 (例如输入 8 或 1.5): ")
            try:
                HOURS_THRESHOLD = float(threshold_input)
                if HOURS_THRESHOLD > 0:
                    break
                else:
                    print("❌ 无效输入，请输入一个正数。")
            except ValueError:
                print("❌ 无效输入，请输入一个有效的数字 (整数或小数)。")

        print(f"\n🔍 正在查找运行超过 {HOURS_THRESHOLD} 小时的活动作业...")

        all_jobs = list(client.batches.list())
        now = datetime.now(timezone.utc)

        long_running_jobs = []
        for job in all_jobs:
            if job.state == genai.types.JobState.JOB_STATE_RUNNING or job.state == genai.types.JobState.JOB_STATE_PENDING:
                create_time = job.create_time.astimezone(timezone.utc)
                duration_hours = (now - create_time).total_seconds() / 3600
                if duration_hours > HOURS_THRESHOLD:
                    job.duration_hours = duration_hours
                    long_running_jobs.append(job)

        if not long_running_jobs:
            print(f"\n✅ 未找到任何运行时间超过 {HOURS_THRESHOLD} 小时的活动作业。")
            return

        print(f"\n" + "-" * 50)
        print(f"🕒 发现 {len(long_running_jobs)} 个超时作业：")
        for i, job in enumerate(long_running_jobs):
            print(f"  {i + 1}. 作业名称: {job.display_name}")
            print(f"     ID: {job.name}")
            print(f"     状态: {job.state.name}")
            print(f"     已运行时长: {job.duration_hours:.2f} 小时")

        print("-" * 50)
        print("⚠️ 警告：此操作将尝试取消并删除以上列出的所有超时作业！")
        confirm = input("您确定要终止所有这些作业吗？请输入 'yes' 以确认: ").lower()

        if confirm == 'yes':
            print("\n🔥 正在终止作业，请稍候...")
            for job in long_running_jobs:
                try:
                    print(f"  - 正在取消作业: {job.display_name}...")
                    client.batches.cancel(name=job.name)
                    print(f"    - 取消成功。")

                    try:
                        client.batches.delete(name=job.name)
                        print(f"    - 已从列表中删除。")
                    except exceptions.PermissionDenied as e:
                        print(f"    - 提示：作业已取消，但立即删除失败 (这通常是正常的，稍后会自动清理): {e}")

                except Exception as e:
                    print(f"  - 🔥 终止作业 {job.display_name} 失败: {e}")
            print("\n✅ 作业终止操作完成！")
        else:
            print("\n🚫 操作已取消。")

    except Exception as e:
        print(f"🔥 获取或管理作业时发生错误: {e}")


# ================================
# 4. 查看最近批处理作业详情
# ================================
def list_recent_batch_jobs(client):
    """
    列出最近的批处理作业，包括内联作业和基于文件的作业详情。
    """
    print("\n" + "=" * 50)
    print("📋 最近批处理作业列表")
    print("=" * 50)
    print("🔍 正在获取最近 10 个批处理作业...\n")

    try:
        batches = client.batches.list(config={'page_size': 10})

        job_count = 0
        for b in batches.page:
            job_count += 1
            print(f"作业名称: {b.name}")
            print(f"  - 显示名称: {b.display_name}")
            print(f"  - 状态: {b.state.name}")
            print(f"  - 创建时间: {b.create_time.strftime('%Y-%m-%d %H:%M:%S')}")

            # 检查是否为内联作业（无目标文件）
            if b.dest is not None:
                if not b.dest.file_name:
                    full_job = client.batches.get(name=b.name)
                    if full_job.inlined_responses:
                        print(f"  - 类型: 内联作业 ({len(full_job.inlined_responses)} 个响应)")
                else:
                    print(f"  - 类型: 基于文件 (输出: {b.dest.file_name})")

            print("-" * 20)

        if job_count == 0:
            print("✅ 未找到任何批处理作业。")
        else:
            print(f"\n✅ 共显示 {job_count} 个批处理作业。")

    except Exception as e:
        print(f"🔥 获取批处理作业列表时发生错误: {e}")

# ================================
# 5.删除指定批处理作业
# ================================

def cancel_batch_job(client):
    job_to_cancel_name = input("请输入要取消的批处理作业名称（如 batches/your-job-name-here）: ").strip()
    try:
        print(f"正在尝试取消作业: {job_to_cancel_name}")
        client.batches.cancel(name=job_to_cancel_name)
        print("作业取消请求已发送。")
    except Exception as e:
        print(f"取消作业时出错: {e}")

# ================================
# 主程序入口
# ================================
def main():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ 错误: 请设置 GEMINI_API_KEY 环境变量")
        return

    try:
        client = genai.Client(api_key=api_key)
        print("✅ Gemini 客户端初始化成功。")
    except Exception as e:
        print(f"❌ Gemini 初始化失败: {e}")
        return

    while True:
        print("\n" + "=" * 50)
        print("🛠️ Gemini 云端资源管理器 🛠️")
        print("=" * 50)
        print("1. 管理已上传的文件 (不含作业结果)")
        print("2. 管理作业结果文件 (列出和批量删除)")
        print("3. 管理批处理作业 (查找并终止超时作业)")
        print("4. 查看最近批处理作业详情")
        print("5. 取消指定批处理作业")
        print("6. 退出")
        choice = input("请输入您的选择 (1, 2, 3, 4, 5, 或 6): ")

        if choice == '1':
            manage_uploaded_files(client)
        elif choice == '2':
            manage_job_result_files(client)
        elif choice == '3':
            manage_batch_jobs(client)
        elif choice == '4':
            list_recent_batch_jobs(client)
        elif choice == '5':
            cancel_batch_job(client)
        elif choice == '6':
            print("👋 再见！")
            break
        else:
            print("❌ 无效选择，请输入 1, 2, 3, 4, 5 或 6。")


if __name__ == "__main__":
    main()