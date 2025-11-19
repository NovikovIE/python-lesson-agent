import subprocess
import sys
from langchain.tools import tool


@tool
def code_executor_tool(code: str) -> str:
    """
    Выполняет переданный Python код и возвращает результат его выполнения.
    Принимает на вход строку с кодом.
    Возвращает stdout и stderr. Используй этот инструмент для проверки
    работоспособности кода или для запуска тестов.
    """
    try:
        process = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        output = ""
        if process.stdout:
            output += f"STDOUT:\n{process.stdout}\n"
        if process.stderr:
            output += f"STDERR:\n{process.stderr}\n"
            
        return output if output else "Код выполнен успешно, вывода нет."

    except subprocess.TimeoutExpired:
        return "Ошибка: Время выполнения кода превысило 30 секунд."
    except Exception as e:
        return f"Ошибка при выполнении кода: {e}"
