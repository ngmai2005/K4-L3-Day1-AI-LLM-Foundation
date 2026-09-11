"""
K4 - Ngày 1: Khám Phá LLM API
Phiên bản dùng Google Gemini API.

Lưu ý:
- call_openai và call_openai_mini được giữ nguyên tên
  để tương thích với bộ test của đề.
- Thực tế hai hàm gọi hai model Gemini khác nhau.
"""

import os
import time
from typing import Callable, Iterable, Optional

from google import genai
from google.genai import types


# ============================================================
# CẤU HÌNH MODEL
# ============================================================

# Có thể thay đổi bằng biến môi trường.
GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash",
)

GEMINI_MINI_MODEL = os.getenv(
    "GEMINI_MINI_MODEL",
    "gemini-2.5-flash-lite",
)

# Giữ tên biến cũ để tương thích nếu template hoặc test có dùng.
OPENAI_MODEL = GEMINI_MODEL
OPENAI_MINI_MODEL = GEMINI_MINI_MODEL


# Giá tham khảo theo 1 triệu token.
# Đây là giá cấu hình để phục vụ bài thực hành ước lượng chi phí.
# Không dùng giá này như hóa đơn thực tế.
MODEL_PRICING = {
    GEMINI_MODEL: {
        "input": 0.30,
        "output": 2.50,
    },
    GEMINI_MINI_MODEL: {
        "input": 0.10,
        "output": 0.40,
    },

    # Alias tương thích với đề/test gốc dùng tên GPT.
    "gpt-4o": {
        "input": 0.30,
        "output": 2.50,
    },
    "gpt-4o-mini": {
        "input": 0.10,
        "output": 0.40,
    },
}


# ============================================================
# GEMINI CLIENT
# ============================================================

def _get_gemini_client():
    """
    Tạo Gemini client.

    Không tạo client ở cấp module để bộ test có thể mock
    google.genai.Client.
    """
    api_key = os.getenv("GEMINI_API_KEY")

    if api_key:
        return genai.Client(api_key=api_key)

    return genai.Client()


# ============================================================
# HÀM HỖ TRỢ
# ============================================================

def _safe_response_text(response) -> str:
    """
    Lấy nội dung text an toàn từ Gemini response.
    """
    text = getattr(response, "text", None)

    if text is None:
        return ""

    return str(text).strip()


def _build_config(
    temperature: float = 0.7,
    system_instruction: Optional[str] = None,
):
    """
    Tạo cấu hình cho Gemini.
    """
    kwargs = {
        "temperature": temperature,
    }

    if system_instruction is not None:
        kwargs["system_instruction"] = system_instruction

    return types.GenerateContentConfig(**kwargs)


# ============================================================
# PART 1 - API CƠ BẢN
# ============================================================

def call_openai(
    prompt: str,
    temperature: float = 0.7,
):
    """
    Gọi model Gemini chính.

    Giữ tên call_openai để tương thích với template và test.
    Trả về:
        (response_text, latency)
    """
    client = _get_gemini_client()

    start_time = time.perf_counter()

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=_build_config(temperature=temperature),
    )

    latency = time.perf_counter() - start_time

    # Tránh latency bằng 0 trong môi trường mock/test.
    latency = max(float(latency), 1e-9)

    return _safe_response_text(response), float(latency)


def call_openai_mini(
    prompt: str,
    temperature: float = 0.7,
):
    """
    Gọi model Gemini mini.

    Đây là model thứ hai, khác với GEMINI_MODEL.
    Trả về:
        (response_text, latency)
    """
    client = _get_gemini_client()

    start_time = time.perf_counter()

    response = client.models.generate_content(
        model=GEMINI_MINI_MODEL,
        contents=prompt,
        config=_build_config(temperature=temperature),
    )

    latency = time.perf_counter() - start_time
    latency = max(float(latency), 1e-9)

    return _safe_response_text(response), float(latency)


def compare_models(prompt: str) -> dict:
    """
    So sánh hai model Gemini trên cùng một prompt.

    Giữ các key gpt4o_* theo yêu cầu của bộ test gốc.
    """
    main_response, main_latency = call_openai(prompt)
    mini_response, mini_latency = call_openai_mini(prompt)

    main_cost = estimate_cost(
        input_text=prompt,
        output_text=main_response,
        model="gpt-4o",
    )

    mini_cost = estimate_cost(
        input_text=prompt,
        output_text=mini_response,
        model="gpt-4o-mini",
    )

    return {
        # Key cũ để tương thích với đề.
        "gpt4o_response": main_response,
        "mini_response": mini_response,
        "gpt4o_latency": main_latency,
        "mini_latency": mini_latency,
        "gpt4o_cost_estimate": main_cost["total_cost"],

        # Thông tin bổ sung giúp hiểu rõ đang dùng Gemini.
        "main_model": GEMINI_MODEL,
        "mini_model": GEMINI_MINI_MODEL,
        "mini_cost_estimate": mini_cost["total_cost"],
        "cost_difference": (
            main_cost["total_cost"] - mini_cost["total_cost"]
        ),
    }


# ============================================================
# PART 2 - SYSTEM PROMPT VÀ TOKEN
# ============================================================

def chat_with_system_prompt(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.7,
):
    """
    Gửi system prompt thông qua config.system_instruction.

    Trả về:
        (response_text, latency)
    """
    client = _get_gemini_client()

    start_time = time.perf_counter()

    config = _build_config(
        temperature=temperature,
        system_instruction=system_prompt,
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_prompt,
        config=config,
    )

    latency = time.perf_counter() - start_time
    latency = max(float(latency), 1e-9)

    return _safe_response_text(response), float(latency)


def count_tokens(text: str) -> int:
    """
    Ước lượng số token.

    Không gọi API để đếm token vì bài test phải chạy được
    ngay cả khi không có API key.

    Công thức gần đúng:
        số token ≈ số từ / 0.75

    Luôn trả về số nguyên >= 1.
    """
    if text is None:
        text = ""

    text = str(text).strip()

    if not text:
        return 1

    words = text.split()
    estimated_tokens = round(len(words) / 0.75)

    return max(1, int(estimated_tokens))


def estimate_cost(
    input_text: str,
    output_text: str,
    model: str = "gpt-4o",
) -> dict:
    """
    Ước lượng chi phí dựa trên số token.

    Chi phí được tính theo đơn vị USD trên 1 triệu token.
    """
    input_tokens = count_tokens(input_text)
    output_tokens = count_tokens(output_text)

    pricing = MODEL_PRICING.get(model)

    if pricing is None:
        # Nếu truyền tên Gemini model chưa có trong bảng,
        # dùng mức giá của model chính hoặc mức mặc định.
        pricing = MODEL_PRICING.get(
            GEMINI_MODEL,
            {
                "input": 0.30,
                "output": 2.50,
            },
        )

    input_cost = (
        input_tokens / 1_000_000
    ) * pricing["input"]

    output_cost = (
        output_tokens / 1_000_000
    ) * pricing["output"]

    total_cost = input_cost + output_cost

    return {
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "input_cost": float(input_cost),
        "output_cost": float(output_cost),
        "total_cost": float(total_cost),
    }


# ============================================================
# PART 3 - STREAMING VÀ RETRY
# ============================================================

def streaming_chatbot():
    """
    Chatbot tương tác bằng Gemini streaming API.

    Nhập quit hoặc exit để kết thúc.
    """
    client = _get_gemini_client()

    print("Gemini Streaming Chatbot")
    print("Nhập 'quit' hoặc 'exit' để kết thúc.")

    while True:
        try:
            user_input = input("\nBạn: ")
        except (EOFError, KeyboardInterrupt):
            print("\nĐã thoát chatbot.")
            break

        if user_input.strip().lower() in {"quit", "exit"}:
            print("Đã thoát chatbot.")
            break

        if not user_input.strip():
            print("Vui lòng nhập nội dung.")
            continue

        config = _build_config(temperature=0.7)

        print("Gemini: ", end="", flush=True)

        try:
            stream = client.models.generate_content_stream(
                model=GEMINI_MODEL,
                contents=user_input,
                config=config,
            )

            for chunk in stream:
                chunk_text = getattr(chunk, "text", None)

                if chunk_text:
                    print(chunk_text, end="", flush=True)

            print()

        except Exception as exc:
            print(f"\nLỗi khi gọi Gemini API: {exc}")


def retry_with_backoff(
    func: Callable,
    max_retries: int = 3,
    base_delay: float = 1.0,
):
    """
    Thực thi func với exponential backoff.

    Ví dụ:
        lần 1 thất bại
        chờ base_delay giây

        lần 2 thất bại
        chờ base_delay * 2 giây

        lần 3 thất bại
        chờ base_delay * 4 giây

    max_retries là tổng số lần thử tối đa.
    """
    if max_retries < 1:
        raise ValueError("max_retries phải >= 1")

    last_exception = None

    for attempt in range(max_retries):
        try:
            return func()

        except Exception as exc:
            last_exception = exc

            # Nếu đã hết số lần thử thì ném lỗi.
            if attempt == max_retries - 1:
                raise

            delay = base_delay * (2 ** attempt)
            time.sleep(delay)

    # Trường hợp dự phòng.
    if last_exception is not None:
        raise last_exception

    raise RuntimeError("Không thể thực thi hàm.")


# ============================================================
# PART 4 - MINI PROJECT ASSISTANT
# ============================================================

def run_assistant(
    persona: str,
    get_input=input,
    max_turns: int = 10,
):
    """
    Chạy trợ lý hội thoại nhiều lượt.

    Tham số:
        persona:
            System prompt của trợ lý.

        get_input:
            Hàm lấy input, mặc định là input.
            Có thể truyền hàm mock khi test.

        max_turns:
            Số lượt hội thoại tối đa.

    Kết quả:
        {
            "num_turns": int,
            "total_tokens": int,
            "total_cost": float,
            "history": list
        }
    """
    history = []
    num_turns = 0
    total_tokens = 0
    total_cost = 0.0

    if max_turns <= 0:
        return {
            "num_turns": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "history": [],
        }

    client = _get_gemini_client()

    print("Mini Assistant")
    print("Nhập 'quit' hoặc 'exit' để kết thúc.")

    while num_turns < max_turns:
        try:
            user_input = get_input("\nBạn: ")
        except (EOFError, KeyboardInterrupt):
            print("\nĐã kết thúc trợ lý.")
            break

        if user_input.strip().lower() in {"quit", "exit"}:
            print("Đã kết thúc trợ lý.")
            break

        if not user_input.strip():
            print("Vui lòng nhập nội dung.")
            continue

        config = _build_config(
            temperature=0.7,
            system_instruction=persona,
        )

        response_parts = []

        try:
            stream = client.models.generate_content_stream(
                model=GEMINI_MODEL,
                contents=user_input,
                config=config,
            )

            print("Trợ lý: ", end="", flush=True)

            for chunk in stream:
                chunk_text = getattr(chunk, "text", None)

                if chunk_text:
                    response_parts.append(str(chunk_text))
                    print(chunk_text, end="", flush=True)

            print()

        except Exception as exc:
            print(f"\nLỗi khi gọi Gemini API: {exc}")
            continue

        assistant_response = "".join(response_parts).strip()

        if not assistant_response:
            assistant_response = "(Trợ lý không trả về nội dung.)"

        # Lưu lịch sử theo từng message.
        history.append(
            {
                "role": "user",
                "content": user_input,
            }
        )

        history.append(
            {
                "role": "assistant",
                "content": assistant_response,
            }
        )

        # Chỉ giữ tối đa 3 lượt = 6 message.
        history = history[-6:]

        turn_tokens = count_tokens(user_input) + count_tokens(
            assistant_response
        )

        turn_cost = estimate_cost(
            input_text=user_input,
            output_text=assistant_response,
            model="gpt-4o",
        )

        total_tokens += turn_tokens
        total_cost += turn_cost["total_cost"]
        num_turns += 1

    return {
        "num_turns": num_turns,
        "total_tokens": int(total_tokens),
        "total_cost": float(total_cost),
        "history": history,
    }


# ============================================================
# HÀM BỔ SUNG
# ============================================================

def batch_compare(prompts: Iterable[str]) -> list:
    """
    So sánh hai model với nhiều prompt.
    """
    results = []

    for prompt in prompts:
        result = compare_models(prompt)
        result["prompt"] = prompt
        results.append(result)

    return results


def format_comparison_table(results: list) -> str:
    """
    Chuyển kết quả compare_models thành bảng text đơn giản.
    """
    if not results:
        return "Không có kết quả."

    lines = [
        "| Prompt | Model chính | Model mini | "
        "Latency chính | Latency mini | Cost chính | Cost mini |",
        "|---|---|---|---:|---:|---:|---:|",
    ]

    for result in results:
        prompt = str(result.get("prompt", "")).replace("|", "\\|")

        main_latency = result.get("gpt4o_latency", 0.0)
        mini_latency = result.get("mini_latency", 0.0)
        main_cost = result.get("gpt4o_cost_estimate", 0.0)
        mini_cost = result.get("mini_cost_estimate", 0.0)

        lines.append(
            f"| {prompt} | "
            f"{result.get('main_model', GEMINI_MODEL)} | "
            f"{result.get('mini_model', GEMINI_MINI_MODEL)} | "
            f"{main_latency:.4f}s | "
            f"{mini_latency:.4f}s | "
            f"${main_cost:.8f} | "
            f"${mini_cost:.8f} |"
        )

    return "\n".join(lines)


# ============================================================
# CHẠY THỬ TRỰC TIẾP
# ============================================================

if __name__ == "__main__":
    print("Hai model đang sử dụng:")
    print(f"- Model chính: {GEMINI_MODEL}")
    print(f"- Model mini:  {GEMINI_MINI_MODEL}")

    print("\nVí dụ so sánh model:")
    prompt = "Hãy kể một sự thật thú vị về Việt Nam."

    try:
        result = compare_models(prompt)

        print("\n--- Model chính ---")
        print(result["gpt4o_response"])

        print("\n--- Model mini ---")
        print(result["mini_response"])

        print("\n--- Thống kê ---")
        print(f"Latency chính: {result['gpt4o_latency']:.4f}s")
        print(f"Latency mini: {result['mini_latency']:.4f}s")
        print(
            f"Chi phí chính: "
            f"${result['gpt4o_cost_estimate']:.8f}"
        )
        print(
            f"Chi phí mini: "
            f"${result['mini_cost_estimate']:.8f}"
        )

    except Exception as exc:
        print(f"Không thể gọi API: {exc}")