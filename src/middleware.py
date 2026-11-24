"""
Middleware агента (задание 8.2).

В langchain 1.x у агента нет хуков pre_ainvoke / post_ainvoke — их роль выполняют
abefore_agent (один раз до начала работы агента) и aafter_agent (один раз после).
Вызовы инструментов перехватывает awrap_tool_call.
"""
import re
import time

from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from loguru import logger

from src.process import content_to_text

MAX_INPUT_LENGTH = 2000

# Что маскируем во входе пользователя, прежде чем текст уйдёт в модель
MASK_RULES = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[EMAIL]"),
    (re.compile(r"\b(?:\d[ -]?){15}\d\b"), "[CARD]"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"), "[API_KEY]"),
]

# Признаки попытки подменить инструкции агента — такой запрос блокируем целиком
INJECTION_PATTERNS = [
    re.compile(r"игнорируй\s+(все\s+)?(предыдущие|прошлые)\s+инструкции", re.IGNORECASE),
    re.compile(r"ignore\s+(all\s+)?(previous|prior)\s+instructions", re.IGNORECASE),
    re.compile(r"(покажи|выведи|раскрой)\s+(свой\s+)?(system|системный)\s*(prompt|промпт)", re.IGNORECASE),
]


class InputFilterMiddleware(AgentMiddleware):
    """Проверяет и очищает вход пользователя до того, как он попадёт в модель."""

    @hook_config(can_jump_to=["end"])
    async def abefore_agent(self, state, runtime):
        last = state["messages"][-1]
        if not isinstance(last, HumanMessage):
            return None
        text = content_to_text(last.content)

        reason = None
        if not text.strip():
            reason = "Пустой запрос. Напишите, что нужно сделать."
        elif len(text) > MAX_INPUT_LENGTH:
            reason = f"Запрос слишком длинный: допускается не более {MAX_INPUT_LENGTH} символов."
        elif any(pattern.search(text) for pattern in INJECTION_PATTERNS):
            reason = "Запрос отклонён: он похож на попытку изменить инструкции ассистента."

        if reason:
            logger.warning(f"[filter] запрос заблокирован: {reason}")
            # jump_to="end" завершает работу агента без обращения к модели
            return {"messages": [AIMessage(content=reason)], "jump_to": "end"}

        masked = text
        for pattern, placeholder in MASK_RULES:
            masked = pattern.sub(placeholder, masked)
        if masked != text:
            logger.warning(f"[filter] во входе скрыты персональные данные: {masked}")
            # сообщение с тем же id заменяет исходное в состоянии агента
            return {"messages": [HumanMessage(content=masked, id=last.id)]}
        return None


class LoggingMiddleware(AgentMiddleware):
    """Логирует вход пользователя, вызовы инструментов и итоговый ответ агента."""

    def __init__(self):
        super().__init__()
        self.started_at = None

    async def abefore_agent(self, state, runtime):
        self.started_at = time.perf_counter()
        logger.info(f"[agent] вход пользователя: {content_to_text(state['messages'][-1].content)}")
        return None

    async def awrap_tool_call(self, request, handler):
        name, args = request.tool_call["name"], request.tool_call["args"]
        logger.info(f"[tool] вызов {name}({args})")
        started = time.perf_counter()
        result = await handler(request)
        elapsed = time.perf_counter() - started
        if isinstance(result, ToolMessage):
            logger.info(f"[tool] {name} -> {result.status} за {elapsed:.2f} c: "
                        f"{content_to_text(result.content)[:300]}")
        return result

    async def aafter_agent(self, state, runtime):
        elapsed = time.perf_counter() - self.started_at if self.started_at else 0.0
        logger.info(f"[agent] ответ за {elapsed:.2f} c: {content_to_text(state['messages'][-1].content)}")
        return None


class ToolErrorMiddleware(AgentMiddleware):
    """
    Превращает исключение инструмента в ToolMessage со статусом error.
    Без этого ошибка MCP-инструмента (например, «город не найден») обрывает работу агента,
    а так модель видит текст ошибки и может объяснить её пользователю.
    """

    async def awrap_tool_call(self, request, handler):
        try:
            return await handler(request)
        except Exception as error:
            logger.error(f"[tool] {request.tool_call['name']} завершился ошибкой: {error}")
            return ToolMessage(
                content=f"Ошибка инструмента: {error}",
                name=request.tool_call["name"],
                tool_call_id=request.tool_call["id"],
                status="error",
            )
